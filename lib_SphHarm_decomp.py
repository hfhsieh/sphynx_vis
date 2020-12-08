#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    The class `SphHarm_decomp` that compute spherical harmonic decomposition
#    of shell with density in certain range.
#
#  Reference:
#    Burrows+, 2012, ApJ, 759, 5
#
#  Last Updated: 2020/12/04
#  He-Feng Hsieh


import operator
import numpy as np
from functools import reduce
from scipy.special import lpmv


class SphHarm_decomp():
    """
    Apply spherical harmonic decomposition and calculate the coefficients
    """
    def __init__(self):
        self.Nlm = dict()

    def cart2sph_pos(self, x, y, z):
        # convert position from Cartesian to spherical coordinate
        r_xy_2  = x * x + y * y
        r_xyz_2 = r_xy_2 + z * z
        r_xy    = np.sqrt(r_xy_2)
        r_xyz   = np.sqrt(r_xyz_2)

        r     = r_xyz
        theta = np.arctan2(r_xy, z)
        phi   = np.arctan2(y, x)

        return r, theta, phi

    def calc_Nlm(self, l, m):
        """
           m          2 * l + 1   (l - m)!
          N  = sqrt( ----------- ---------- )
           l            4 * pi    (l + m)!
        """
        Nlm = self.Nlm.get((l, m), None)

        if Nlm is None:
            Nlm_sq = (2 * l + 1) / (4 * np.pi) / reduce(operator.mul, range(l - m + 1, l + m + 1), 1)
            Nlm = np.sqrt(Nlm_sq)

            self.Nlm[(l, m)] = Nlm

        return Nlm

    def calc_alm(self, l, m, phi, theta):
        """
        Calculate the spherical harmonic component a_{lm}.

        Caution: the result from sympy and scipy could differ by a factor of -1
                 if m is negative and -m is odd
        """
        if not isinstance(phi, np.ndarray):
            phi = np.array(phi)

        if not isinstance(theta, np.ndarray):
            theta = np.array(theta)

        m_abs = np.abs(m)
        norm_fac = (-1.)**m_abs / np.sqrt(4 * np.pi * (2 * l + 1))

        Nlm = self.calc_Nlm(l, m_abs)
        Plm = lpmv(m_abs, l, np.cos(theta))

        if m > 0:
            ylm = np.sqrt(2) * Nlm * Plm * np.cos(m_abs * phi)
        elif m < 0:
            # Note the term (-1)^m is not included in Burrows et al., 2012, ApJ, 759, 5
            ylm = (-1)**m * np.sqrt(2) * Nlm * Plm * np.sin(m_abs * phi)
        else:
            ylm = Nlm * Plm

        return norm_fac * ylm

    def calc_sphharm_decomp(self, binary, dens, ls, ms, fraction = 0.01):
        """
        Decompose the shell with density: [dens * (1 - fraction), dens * (1 + fraction)]
        into spherical harmonic components, and calculate the associated coefficients, a_{lm}.

        Parameters
        ----------
        binary: SPHYNX's binary output (numpy.array)
        """
        if not isinstance(ls, (tuple, list)):
            ls = ls,

        if not isinstance(ms, (tuple, list)):
            ms = ms,

        for l, m in zip(ls, ms):
            assert np.abs(m) <= l, "Invalid (l, m): ({}, {})".format(l, m)

        ### data selection
        dmin = dens * (1 - fraction)
        dmax = dens * (1 + fraction)

        idx = np.where((dmin <= binary["promro"]) & (binary["promro"] <= dmax))[0]
#        print("Number of particles in the shell: {}".format(idx.size))

        radius, theta, phi = self.cart2sph_pos(binary["x"][idx],
                                               binary["y"][idx],
                                               binary["z"][idx] )

        dr_left_index  = np.maximum(idx - 1, 0)
        dr_right_index = np.minimum(idx + 1, binary["radius"].size - 1)

        dr    = 0.5 * (binary["radius"][dr_right_index] - binary["radius"][dr_left_index])
        vol   = 4 * np.pi * dr * radius**2
        r_vol = radius * vol

        weight = vol.sum()

        ### calculate the coefficients
        alm_all = dict()

        for l, m in zip(ls, ms):
            alm = self.calc_alm(l, m, phi, theta) * r_vol

            alm_all[(l, m)] = alm.sum() / weight

        return alm_all

    def get_sphharm_decomp_sphynx(self, data, dens, ls, ms, fraction = 0.01):
        # compute anisotropic velocity using binary data with binning
        return self.calc_sphharm_decomp(data, dens, ls, ms, fraction = fraction)

    def get_sphharm_decomp_yt(self, data, **kwargs):
        raise ValueError ("Not supported!")

