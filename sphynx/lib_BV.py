#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    The class `BV_frequency` that compute the Brunt-Vaisala frequency
#
#    Adopted from Kuo-Chuan's code `get_bv_frequency.py`, with modification for SPHYNX's results
#
#  Reference:
#    See Eqs. (12) and (13) in Pan+, 2016, ApJ, 817, 72.
#
#    Note that -1 is missing in Eq. (13).
#
#  Last Updated: 2020/11/29
#  He-Feng Hsieh


import yt
import numpy as np
from bisect import bisect_right
from .lib_radprof import RadialProfile
from .lib_kernels import *


# physcial constant
const_G = 6.67428e-8

dict_kernel = {1: Kernel_CubicSpline(),
               2: Kernel_Harmonic(),
               3: Kernel_InteHarmonic() }  # indice is adopted from parameters.f90 in SPHYNX


class BV_frequency():
    """
    Calculate the Brunt-Vaisala frequency
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

    def calc_bv_frequency_bin(self, quant_key, ds_or_binary, dr = 1e5, rmax = 2e7, rsh = None):
        """
        calculate the Brunt-Vaisala frequency using binned data

        Parameters
        ----------
        quant_key: sequence of string that specified the keyword to the required quantities
                   in order of [density, entropy, ye]
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

        field_dens, field_entr, field_ye = quant_key

        xradius = RadProf.radius
        xdens   = RadProf.profiles[field_dens]
        xentr   = RadProf.profiles[field_entr]
        xye     = RadProf.profiles[field_ye  ]
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

        dsdr[0]    =       (xentr[1 ] - xentr[0  ]) / dr
        dsdr[1:-1] = 0.5 * (xentr[2:] - xentr[:-2]) / dr
        dsdr[-1]   =       (xentr[-1] - xentr[ -2]) / dr

        dydr[0]    =       (xye[1 ] - xye[0  ]) / dr
        dydr[1:-1] = 0.5 * (xye[2:] - xye[:-2]) / dr
        dydr[-1]   =       (xye[-1] - xye[ -2]) / dr

        mass   = 4.0 * np.pi * dr * np.cumsum(xdens * xradius**2)
        dphidr = -const_G * mass / xradius**2

        ### calculate the BV frequency
        cL = -drhodp * (dpds * dsdr + dpdye * dydr)
        xwbv = np.where(xradius <= rsh,
                        -1.0 * np.sign(cL) * np.sqrt(np.abs(cL / xdens * dphidr)) * 1.e-3,
                        np.NaN)

        return xradius, xwbv

    def calc_bv_frequency(self, binary, rmax = 2e7, rsh = None,
                          formalism = "sph", kernel = None):
        """
        Same as calc_bv_frequency_bin(), but use the binary data without binning

        Parameters
        ----------
        formalism: "sph" or None. Specify which method is used to compute the radial gradient of s and Ye.
                   If None, the forward euler method is used.
                   If "sph", use SPH formalism.
        kernel: SPH kernel
        """
        assert formalism in ["sph", None], "Unknown formalism: {}.".format(formalism)

        if rsh is None:
            rsh = rmax

        xradius = binary["radius"]
        xdens   = binary["promro"]
        xentr   = binary["s"]
        xye     = binary["ye"]

        ### calculate the derivative
        if formalism is None:
            dr   = np.diff(xradius)
            dsdr = np.diff(xentr) / dr
            dydr = np.diff(xye)   / dr

            # for forward euler case, we compute the bv at the center between particles
            xradius = 0.5 * (xradius[1:] + xradius[:-1])
            xdens   = 0.5 * (xdens  [1:] + xdens  [:-1])
            xentr   = 0.5 * (xentr  [1:] + xentr  [:-1])
            xye     = 0.5 * (xye    [1:] + xye    [:-1])

            dr   = np.diff(xradius, prepend = 0)
            mass = 4.0 * np.pi * dr * np.cumsum(xdens * xradius**2)
        else:
            x    = binary["x"]
            y    = binary["y"]
            z    = binary["z"]
            h    = binary["h"]
            mass = binary["mass_cum"]
            parmass = np.diff(mass, prepend = 0)

            # trim the data to save time for calculating gradient
            idx = bisect_right(xradius, rsh)
            rsph_max = np.max(xradius[:idx] + 2 * h[:idx])  # maximum radius of particle needed

            idx = bisect_right(xradius, rsph_max)
            xradius = xradius[:idx]
            xdens   = xdens[:idx]
            xentr   = xentr[:idx]
            xye     = xye[:idx]
            x       = x[:idx]
            y       = y[:idx]
            z       = z[:idx]
            h       = h[:idx]
            mass    = mass[:idx]
            parmass = parmass[:idx]

            dsdr = calc_sph_radgrad_kdtree(xentr, x, y, z, xradius, h, xdens, parmass, kernel)
            dydr = calc_sph_radgrad_kdtree(xye,   x, y, z, xradius, h, xdens, parmass, kernel)

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


        ### calculate the BV frequency
        cL = -drhodp * (dpds * dsdr + dpdye * dydr)
        xwbv = np.where(xradius <= rsh,
                        -1.0 * np.sign(cL) * np.sqrt(np.abs(cL / xdens * dphidr)) * 1.e-3,
                        np.NaN)

        return xradius, xwbv

    def get_bv_sphynx_direct(self, data, rmax = 2e7, rsh = None):
        # compute bv using binary data without binning and the euler method for graident
        #
        # Caution: this method is inaccurate, should never be used!!
        return self.calc_bv_frequency(data, rmax = rmax, rsh = rsh, formalism = None)

    def get_bv_sphynx_sph(self, data, dr = 2e5, rmax = 2e7, rsh = None, kernel_idx = 3, bin_data = False):
        # compute bv using binary data without binning and the SPH formalism for graident
        kernel = dict_kernel[kernel_idx]

        r, bv = self.calc_bv_frequency(data, rmax = rmax, rsh = rsh, formalism = "sph", kernel = kernel)

        if bin_data:
            quant_key = "radius", "bv"
            data      = {"radius": r, "bv": bv}

            RadProf = RadialProfile(dr = dr, rmax = rmax)
            RadProf.get_profiles(quant_key, binary = data, check = False)

            r  = RadProf.profiles["radius"]
            bv = RadProf.profiles["bv"]

        return r, bv

    def get_bv_sphynx_bin(self, data, dr = 1e5, rmax = 2e7, rsh = None):
        # compute bv using binary data with binning
        quant_key = "promro", "s", "ye"

        return self.calc_bv_frequency_bin(quant_key, data, dr = dr, rmax = rmax, rsh = rsh)

    def get_bv_yt(self, data, dr = 1e5, rmax = 2e7, rsh = None):
        # compute bv using re-sampling meshgrid data via yt
        quant_key = [('deposit', 'all_smoothed_particle_gas_density'),
                     ('deposit', 'all_smoothed_s'),
                     ('deposit', 'all_smoothed_ye') ]

        return self.calc_bv_frequency_bin(quant_key, data, dr = dr, rmax = rmax, rsh = rsh)

