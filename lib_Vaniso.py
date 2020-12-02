#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    The class `Vaniso` that compute the anisotropic velocity:
#
#      v_aniso^2 = < rho * \sum_i (v_i - v_i_bar)^2 > / < rho >
#
#    where <x> is the spherically averaged of quantity x
#          v_i        velocity
#          v_i        < v_i >
#
#  Reference:
#    See, for instance, Eq. (14) in Pan+, 2016, ApJ, 817, 72.
#
#  Last Updated: 2020/12/02
#  He-Feng Hsieh


import yt
import numpy as np
from bisect import bisect_right
from scipy.interpolate import interp1d
from .lib_radprof import RadialProfile


# physcial constant
const_c = 2.99792458e10  # speed of light


class Vaniso():
    """
    Calculate the anisotropic velocity
    """
    def __init__(self):
        pass

    def cart2sph_vel(self, x, y, z, vx, vy, vz):
        # convert velocity from Cartesian to spherical coordinate
        r_xy_2  = x * x + y * y
        r_xyz_2 = r_xy_2 + z * z
        r_xy    = np.sqrt(r_xy_2)
        r_xyz   = np.sqrt(r_xyz_2)

        r     = r_xyz
        theta = np.arctan2(r_xy, z)
        phi   = np.arctan2(y, x)

        v_r     = (x * vx + y * vy + z * vz) / r_xyz
        v_theta = (z * (x * vx + y * vy) - r_xy_2 * vz) / (r_xyz_2 * r_xy)
        v_phi   = (vx * y - x * vy) / r_xy_2

        return r, theta, phi, v_r, v_theta, v_phi

    def calc_vaniso_bin(self, ds_or_binary, dr = 1e5, rmax = 2e7, rsh = None,
                        include_vphi = False, include_vphi_ave = False,
                        include_vtheta = False, include_vtheta_ave = False):
        """
        Calculate the anisotropic velocity using binned binary data

        Parameters
        ----------
        ds_or_binary: only support SPHYNX's binary output (numpy.array) currently
        dr, rmax: see RadialProfile class in get_profiles.py
        rsh: shock radius. Set to rmax if not specified
        include_vphi:       Boolean. If True, v_phi       is included in v_aniso
        include_vtheta:     Boolean. If True, v_theta     is included in v_aniso
        include_vphi_ave:   Boolean. If True, < v_phi >   is included in v_aniso
                            Do nothing if include_vphi is False
        include_vtheta_ave: Boolean. If True, < v_theta > is included in v_aniso
                            Do nothing if include_vtheta is False
        """
        if rsh is None:
            rsh = rmax

        ### create a new container for required quantities
        quant_key = "promro", "vr", "vtheta", "vphi"
        data = dict()

        data["radius"] = ds_or_binary["radius"]  # density
        data["promro"] = ds_or_binary["promro"]  # density
        _, _, _, v_r, v_theta, v_phi = self.cart2sph_vel(ds_or_binary[ "x"],
                                                         ds_or_binary[ "y"],
                                                         ds_or_binary[ "z"],
                                                         ds_or_binary["vx"],
                                                         ds_or_binary["vy"],
                                                         ds_or_binary["vz"], )
        data["vr"]     = v_r
        data["vtheta"] = v_theta
        data["vphi"]   = v_phi

        ### calculate the spherically averaged density and velocities
        field_dens, field_vr, field_vt, filed_vp = quant_key

        RadProf = RadialProfile(dr = dr, rmax = rmax)
        RadProf.get_profiles(quant_key, binary = data, check = False)

        radius     = RadProf.radius
        dens_ave   = RadProf.profiles[field_dens]
        vr_ave     = RadProf.profiles[field_vr  ]
        vtheta_ave = RadProf.profiles[field_vt  ]
        vphi_ave   = RadProf.profiles[filed_vp  ]

        ### calculate the numerator: variance = rho * \sum_i (v_i - v_i_bar)^2
        # use interpolation to approximate the averaged velocity where the particles locate
        inte_vr = interp1d(radius, vr_ave,     fill_value = "extrapolate")
        inte_vt = interp1d(radius, vtheta_ave, fill_value = "extrapolate")
        inte_vp = interp1d(radius, vphi_ave,   fill_value = "extrapolate")

        variance = ( data["vr"] - inte_vr( ds_or_binary["radius"] ) )**2

        if include_vtheta:
            if include_vtheta_ave:
                variance += ( data["vtheta"] - inte_vt( ds_or_binary["radius"] ) )**2
            else:
                variance += ( data["vtheta"] )**2

        if include_vphi:
            if include_vphi_ave:
                variance += ( data["vphi"] - inte_vp( ds_or_binary["radius"] ) )**2
            else:
                variance += ( data["vphi"] )**2

        data["variance"] = variance * data["promro"]

        # calculate the averaged variance
        RadProf.get_profiles("variance", binary = data)
        variance_ave = RadProf.profiles["variance"]

        ### calculate the anisotropic velocity
        v_aniso = np.sqrt(variance_ave / dens_ave)
        v_aniso /= const_c  # in units of light speed

        return radius, v_aniso

    def get_vaniso_sphynx_bin(self, data, **kwargs):
        # compute anisotropic velocity using binary data with binning
        return self.calc_vaniso_bin(data, **kwargs)

    def get_vaniso_yt(self, data, **kwargs):
        raise ValueError ("Not supported!")

