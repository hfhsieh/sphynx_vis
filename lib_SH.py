#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    The class `SH_instability` that compute the Solberg-Hoiland criterion, R_SH
#
#    When R_SH >= 0: stable in the vertical direction
#         R_SH <  0: unstable
#
#    Note that the first term in Solberg-Hoiland criterion is the Ledoux criterion,
#    which is equivalent to in the cases of nuclear EoS:
#
#        (d\rho / dr)_adia - (d\rho / dr)
#      = (∂\rho / ∂P)_{s, Y_l} * [ (∂P / ∂s  )_{\rho, Y_l} * (ds   / dr)
#                                + (∂P / ∂Y_l)_{\rho, s  } * (dY_l / dr) ]
#      = -C_L
#
#    where Y_l = Y_e + Y_nu + Y_nubar is the abundance of lepton.
#
#
#  Reference:
#    See Eqs. (23) and (24) in Heger+, 2000, ApJ, 528, 368
#
#  Last Updated: 2020/12/01
#  He-Feng Hsieh


import yt
import numpy as np
from bisect import bisect_right
from .lib_radprof import RadialProfile
from .lib_kernels import *


# physcial constant
const_G = 6.67428e-8
A_alpha = 4.0        # mass number of alpha particle (in amu)
A_n = 1.00866491588  # mass number of neutron (in amu)
A_p = 1.00727646662  # mass number of proton (in amu)


dict_kernel = {1: Kernel_CubicSpline(),
               2: Kernel_Harmonic(),
               3: Kernel_InteHarmonic() }  # indice is adopted from parameters.f90 in SPHYNX


class SH_criterion():
    """
    Calculate the Solberg-Hoiland criterion
    """
    def __init__(self, neos):
        self.neos = neos  # a NuclearEOS object defined in NuclearEos
        self._get_neos_boundary()

    def _get_neos_boundary(self):
        brd = self.neos.get_boundaries()

        self.dens_min = brd[0]
        self.dens_max = brd[1]
        self.temp_min = brd[2]
        self.temp_max = brd[3]
        self.ye_min   = brd[4]
        self.ye_max   = brd[5]

    def _check_neos(self, dens = None, temp = None, ye = None):
        """
        check if the given density, temperature, and Ye are in the domain of EoS table
        """
        if dens is not None:
            assert dens > self.dens_min, "density is too low: {}".format(dens)
            assert dens < self.dens_max, "density is too high: {}".format(dens)

        if temp is not None:
            assert temp > self.temp_min, "temperature is too low: {}".format(temp)
            assert temp < self.temp_max, "temperature is too high: {}".format(temp)

        if ye   is not None:
            assert ye   > self.ye_min,   "Ye is too low: {}".format(ye)
            assert ye   < self.ye_max,   "Ye is too high: {}".format(ye)

    def _calc_mu(self):
        """
        calculate the mean molecular weight, mu

            m_u = 1 / sum_i(Y_i) = 1 / sum_i(X_i / A_i)

        where Y_i = X_i / A_i is abundance of species i
              X_i             is mass fraction of species i
              A_i = m_i / m_u is the mass number of species i
              m_u             is atomic mass unit

        For instance, Y_e = sum_i(Z_i * Y_i) is the electron abundance

        For nuclear EoS, we compute m_u by considering four species:

            alpha particle, heavy nucleus, neutron, and proton

        i.e.
            m_u = 1 / (X_alpha / A_alpha + X_h / A_bar + X_n / A_n + X_p / A_p)
        """
        # we assume neos.nuc_eos_full() is called
        return 1.0 / ( self.neos.xxa / A_alpha
                     + self.neos.xxh / self.neos.xabar
                     + self.neos.xxn / A_n
                     + self.neos.xxp / A_p )

    def _get_drhodp(self, d, s, ye, fratio = 0.01):
        d1 = (1 - fratio) * d
        d2 = (1 + fratio) * d

        self._check_neos(dens = d1, ye = ye)
        self._check_neos(dens = d2, ye = ye)

        v1 = self.neos.getEOSfromRhoEntrYe(d1, s, ye)
        v2 = self.neos.getEOSfromRhoEntrYe(d2, s, ye)

        return (d2 - d1) / (v2.xprs - v1.xprs)

    def _get_dpds(self, d, s, ye, fratio = 0.01):
        s1 = (1 - fratio) * s
        s2 = (1 + fratio) * s

        self._check_neos(dens = d, ye = ye)

        v1 = self.neos.getEOSfromRhoEntrYe(d, s1, ye)
        v2 = self.neos.getEOSfromRhoEntrYe(d, s2, ye)

        return (v2.xprs - v1.xprs) / (s2 - s1)

    def _get_dpdye(self, d, s, ye, fratio = 0.01):
        y1 = (1 - fratio) * ye
        y2 = (1 + fratio) * ye

        self._check_neos(dens = d, ye = y1)
        self._check_neos(dens = d, ye = y2)

        v1 = self.neos.getEOSfromRhoEntrYe(d, s, y1)
        v2 = self.neos.getEOSfromRhoEntrYe(d, s, y2)

        return (v2.xprs - v1.xprs) / (y2 - y1)

    def calc_sh_criterion_bin(self, quant_key, ds_or_binary, dr = 1e5, rmax = 2e7, rsh = None):
        """
        Calculate the Solberg-Hoiland criterion using binned data

        Parameters
        ----------
        quant_key: sequence of string that specified the keyword to the required quantities
                   in order of [density, entropy, ye, omega]
        ds_or_binary: yt object or SPHYNX's binary output (numpy.array)
        dr, rmax: see RadialProfile class in get_profiles.py
        rsh: shock radius. Set to rmax if not specified
        """
        if rsh is None:
            rsh = rmax

        ### build the radial profiles of density, entropy, and ye
        RadProf = RadialProfile(dr = dr, rmax = rmax)

        if isinstance(ds_or_binary, np.ndarray):  # from SPHYNX's binary output
            RadProf.get_profiles(quant_key, binary = ds_or_binary)
        else:  # from yt object
            RadProf.get_profiles(quant_key, ds     = ds_or_binary)

        field_dens, field_entr, field_ye, field_omega = quant_key

        xradius = RadProf.radius
        xdens   = RadProf.profiles[field_dens ]
        xentr   = RadProf.profiles[field_entr ]
        xye     = RadProf.profiles[field_ye   ]
        xomega  = RadProf.profiles[field_omega]
        dr      = RadProf.dr

        ### calculate the derivative
        drhodp = np.array( [ self._get_drhodp(d, s, ye)  for d, s, ye in zip(xdens, xentr, xye) ] )
        dpds   = np.array( [ self._get_dpds  (d, s, ye)  for d, s, ye in zip(xdens, xentr, xye) ] )
        dpdye  = np.array( [ self._get_dpdye (d, s, ye)  for d, s, ye in zip(xdens, xentr, xye) ] )

        # Check if we got nan
        check = True

        if check:
            if np.any(np.isnan(drhodp)):
                print("Found nan in drhodp")
            if np.any(np.isnan(dpds)):
                print("Found nan in dpds")
            if np.any(np.isnan(dpdye)):
                print("Found nan in dpdye")

        dsdr = np.empty(xradius.size)
        dydr = np.empty(xradius.size)
        dwdr = np.empty(xradius.size)  # d\omega / dr

        dsdr[0]    =       (xentr[1 ] - xentr[0  ]) / dr
        dsdr[1:-1] = 0.5 * (xentr[2:] - xentr[:-2]) / dr
        dsdr[-1]   =       (xentr[-1] - xentr[ -2]) / dr

        dydr[0]    =       (xye[1 ] - xye[0  ]) / dr
        dydr[1:-1] = 0.5 * (xye[2:] - xye[:-2]) / dr
        dydr[-1]   =       (xye[-1] - xye[ -2]) / dr

        dwdr[0]    =       (xomega[1 ] - xomega[0  ]) / dr
        dwdr[1:-1] = 0.5 * (xomega[2:] - xomega[:-2]) / dr
        dwdr[-1]   =       (xomega[-1] - xomega[ -2]) / dr

        mass   = 4.0 * np.pi * dr * np.cumsum(xdens * xradius**2)
        dphidr = -const_G * mass / xradius**2

        ### calculate the SH criterion
        cL   = -drhodp * (dpds * dsdr + dpdye * dydr)
        R_SH = -1e-6 * cL * dphidr / xdens + 2.0 * xomega * (xradius * dwdr + 2.0 * xomega)  # [ms^-2]

        R_SH = np.where(xradius <= rsh, R_SH, np.NaN)

        return xradius, R_SH

    def calc_sh_criterion(self, binary, rmax = 2e7, rsh = None, kernel = None):
        """
        Same as calc_sh_criterion_bin(), but use the binary data without binning and
        the gradient is obtained using SPH formalism.

        Parameters
        ----------
        kernel: SPH kernel
        """
        if rsh is None:
            rsh = rmax

        ### trim the data to save time for calculating gradient
        # maximum radius of particle needed
        xradius = binary["radius"]
        h       = binary["h"]
        idx = bisect_right(xradius, rsh)

        rsph_max = np.max(xradius[:idx] + 2 * h[:idx])
        idx = bisect_right(xradius, rsph_max)

        xradius = binary["radius"][:idx]
        xomega  = binary["dummy1"][:idx]  # we assume the omega is already calcalated and stored in "dummy1"
        xdens   = binary["promro"][:idx]
        xye     = binary["ye"][:idx]
        xentr   = binary["s"][:idx]
        x       = binary["x"][:idx]
        y       = binary["y"][:idx]
        z       = binary["z"][:idx]
        h       = binary["h"][:idx]
        mass    = binary["mass_cum"][:idx]
        parmass = np.diff(mass, prepend = 0)

        ### calculate the derivative
        dsdr = calc_sph_radgrad_kdtree(xentr,  x, y, z, xradius, h, xdens, parmass, kernel)
        dydr = calc_sph_radgrad_kdtree(xye,    x, y, z, xradius, h, xdens, parmass, kernel)
        dwdr = calc_sph_radgrad_kdtree(xomega, x, y, z, xradius, h, xdens, parmass, kernel)  # d\omega / dr

        dphidr = -const_G * mass / xradius**2

        drhodp = np.array( [ self._get_drhodp(d, s, ye)  for d, s, ye in zip(xdens, xentr, xye) ] )
        dpds   = np.array( [ self._get_dpds  (d, s, ye)  for d, s, ye in zip(xdens, xentr, xye) ] )
        dpdye  = np.array( [ self._get_dpdye (d, s, ye)  for d, s, ye in zip(xdens, xentr, xye) ] )

        # Check if we got nan
        check = True

        if check:
            if np.any(np.isnan(drhodp)):
                print("Found nan in drhodp")
            if np.any(np.isnan(dpds)):
                print("Found nan in dpds")
            if np.any(np.isnan(dpdye)):
                print("Found nan in dpdye")

        ### calculate the SH criterion
        cL   = -drhodp * (dpds * dsdr + dpdye * dydr)
        R_SH = -1e-6 * cL * dphidr / xdens + 2.0 * xomega * (xradius * dwdr + 2.0 * xomega)  # [ms^-2]

        R_SH = np.where(xradius <= rsh, R_SH, np.NaN)

        return xradius, R_SH

    def get_sh_sphynx_sph(self, data, dr = 2e5, rmax = 2e7, rsh = None, kernel_idx = 3, bin_data = False):
        # compute SH criterion using binary data without binning and the SPH formalism for graident
        kernel = dict_kernel[kernel_idx]

        r, sh = self.calc_sh_criterion(data, rmax = rmax, rsh = rsh, kernel = kernel)

        if bin_data:
            quant_key = "radius", "sh"
            data      = {"radius": r, "sh": sh}

            RadProf = RadialProfile(dr = dr, rmax = rmax)
            RadProf.get_profiles(quant_key, binary = data, check = False)

            r  = RadProf.profiles["radius"]
            sh = RadProf.profiles["sh"]

        return r, sh

    def get_sh_sphynx_bin(self, data, dr = 1e5, rmax = 2e7, rsh = None):
        # compute SH criterion using binary data with binning
        # we assume the omega is already calcalated and stored in "dummy1"
        quant_key = "promro", "s", "ye", "dummy1"

        return self.calc_sh_criterion_bin(quant_key, data, dr = dr, rmax = rmax, rsh = rsh)

    def get_sh_yt(self, data, dr = 1e5, rmax = 2e7, rsh = None):
        # compute SH criterion using re-sampling meshgrid data via yt
        quant_key = [('deposit', 'all_smoothed_particle_gas_density'),
                     ('deposit', 'all_smoothed_s'),
                     ('deposit', 'all_smoothed_ye'),
                     ('deposit', 'all_smoothed_omega_phi') ]

        return self.calc_sh_criterion_bin(quant_key, data, dr = dr, rmax = rmax, rsh = rsh)

