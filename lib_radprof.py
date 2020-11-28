#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    The class `RadialProfile` that compute the \radial profiles
#
#    Adopted from Kuo-Chuan's code `get_profiles.py`, with modification for SPHYNX's results
#
#  Last Updated: 2020/11/29
#  He-Feng Hsieh


import numpy as np
from bisect import bisect
from yt import create_profile as yt_create_profile
from scipy.interpolate import interp1d
from .fmt import fmt_binary


# convert field name into set to save time for 'in' operator
quant_binary  = set(fmt_binary.names)


class RadialProfile():
    """
    Get uniform spaced radial profiles from 3D data.
    Note: here we assume center located at [0,0,0] for yt data
    """
    def __init__(self, dr = 1e5, rmax = 5e7):
        self.dr         = dr  # the bin width and the center of lowest bin
        self.rmax       = rmax  # the center of highest bin
        self.nbins      = int(rmax / dr)
        self.radius     = np.linspace(dr, rmax, self.nbins)
        self.radius_brd = np.linspace(0.5 * dr, rmax + 0.5 * dr, self.nbins + 1)  # bin boundary
        self.profiles   = dict()

        self.verbose    = False

    def _get_profile_yt(self, quant, ds):
        # get profiles from yt object
        radius_keyword = ("deposit", "mesh_radius")

        source     = ds.sphere([0, 0, 0], (self.rmax, "cm"))
        yt_profile = yt_create_profile(source,
                                       radius_keyword,
                                       quant,
                                       n_bins = self.nbins,
                                       extrema = {radius_keyword: (self.radius_brd[0],
                                                                   self.radius_brd[-1])},
                                       logs = {radius_keyword: False},
                                       weight_field = ("deposit", "all_smoothed_mass"),
                                       accumulation = False)

        for q in quant:
            self.profiles[q] = yt_profile[q].v

    def _get_profile_binary(self, quant, binary):
        # get profiles from SPHYNX's binary output
        radius = binary["radius"]

        # check whether the radius increases monotonically
        assert np.all(np.diff(radius) > 0), "The radius of particle does not increase monotonically."

        # find indices where the bin boundary should be inserted
        idx_brd = np.searchsorted(radius, self.radius_brd)

        for q in quant:
            data = binary[q]
            prof = [ np.average( data[i:j] )  for i, j in zip( idx_brd, idx_brd[1:] ) ]  # no weighting here

            self.profiles[q] = np.array(prof)

    def get_profiles(self, quant, ds = None, binary = None, check = True):
        """
        Interface for calling the correct function to compute the profiles.

        Parameters
        ----------
        quant: sequence. The list of quantities where the radial profile will be computed
        ds: yt object. If ds is given, the profiles will obtained from the yt object
        binary: data in SPHYNX's binary output
        """
        if ds is None:  # from SPHYNX's binary output
            if check:
                # check if specified quantity is stored in the input binary data
                for q in quant:
                    assert q in quant_binary, "Unknown quantity: {}".format(q)

            self._get_profile_binary(quant, binary)
        else:  # from yt object
            if check:
                # check if specified quantity is stored in the input ds object
                for q in quant:
                    assert q in ds.derived_field_list, "Unknown quantity: {}".format(q)

            self._get_profile_yt(quant, ds)

        # remove empty bins
        self.remove_empty_bins(quant)

    def remove_empty_bins(self, quant):
        # remove empty bins in the profiles by filling the interpolated values
        if self.verbose:
            print("Start removing empty bins")

        for q in quant:
            prof = self.profiles[q]

            is_nonzero = (prof != 0.0)

            if np.all(is_nonzero):
                continue  # all bins are not empty. Nothing to do.
            else:
                print("Warning! found zeros in radial profiles, quantity = {}".format(q))

            # generate the interpolation function
            prof_nonzero = prof[is_nonzero]
            inte_linear  = interp1d(self.radius[is_nonzero], prof_nonzero,
                                    kind = "linear",
                                    bounds_error = False,
                                    fill_value = (prof_nonzero[0], prof_nonzero[-1]))

            # do interpolation
            prof[~is_nonzero]  = inte_linear(self.radius[~is_nonzero])
            self.profiles[var] = prof

        if self.verbose:
            print("Finish removing empty bins")

