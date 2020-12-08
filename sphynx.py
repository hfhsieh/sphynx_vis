#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Class objects that contains functions for visualizing SPHYNX resutls
#
#  Last Updated: 2020/12/04
#  He-Feng Hsieh


import os
import re
import yt
import numpy as np
import matplotlib as mpl
from glob import glob
from bisect import bisect
from subprocess import check_output
from collections.abc import Iterable
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from matplotlib.colors import DivergingNorm
from yt.fields.particle_fields import add_volume_weighted_smoothed_field
from NuclearEos import NuclearEOS
from .fmt import *
from .lib_gw import *
from .lib_kernels import *
from .lib_BV import BV_frequency
from .lib_SH import SH_criterion
from .lib_Vaniso import Vaniso
from .lib_SphHarm_decomp import SphHarm_decomp


# physcial constant
const_G = 6.67428e-8
const_c = 2.99792e10
kpc2cm  = 3.086e21    # kpc in units of cm
sec2ms  = 1e3         # second in units of ms


# convert field name into set to save time for 'in' operator
quant_tiempos = set(fmt_tiempos.names)
quant_central = set(fmt_central_values.names)
quant_lumin   = set(fmt_luminosity.names)
quant_radgrav = set(fmt_radgrav.names)
quant_binary  = set(fmt_binary.names)


class sphynx_ascii():
    """
    Contains functions for loading SPHYNX's ASCII results
    """
    def __init__(self, rundir, runpath):
        self.path = os.path.join(runpath, rundir)

        self.central_values = None
        self.radgrav        = None
        self.luminosity     = dict()
        self.pb             = None
        self.pb_time        = None
        self.pb_nout        = None
        self.time           = None
        self.time_rel       = None  # time relative to bounce

        self.ascii_loader = {"central_values": self.get_central_values,
                             "luminosity":     self.get_luminosity,
                             "radgrav":        self.get_radgrav        }

    def _get_ascii_table(self, key):
        assert key in ["central_values", "luminosity", "radgrav"], "Unknown key: {}".format(key)

        if key == "central_values":
            return self.central_values
        elif key == "luminosity":
            return self.luminosity
        elif key == "radgrav":
            return self.radgrav

    def get_bounce(self):
        # load postbounce time in bounce.d
        try:
            fn = os.path.join(self.path, "bounce.d")
            self.pb_nout, self.pb_time = np.genfromtxt(fn, usecols = [0, 1], skip_header = 1, unpack = True)
            self.pb_nout = np.int(self.pb_nout)
            self.pb      = True
        except OSError:
            self.pb_time = 0.0
            self.pb      = False

    def get_central_values(self):
        fn = os.path.join(self.path, "central_values.d")
        self.central_values = np.genfromtxt(fn, dtype = fmt_central_values)

    def get_radgrav(self):
        fn = os.path.join(self.path, "radgrav.d")
        self.radgrav = np.genfromtxt(fn, dtype = fmt_radgrav)

    def get_tiempos(self):
        fn   = os.path.join(self.path, "tiempos.d")
        data = np.genfromtxt(fn, dtype = fmt_tiempos)

        # if the simulation restarts from postbounce, the relative time is not correct!
        # here we obtain the relative time from the physical time
        self.time     = dict(zip(data["idx"], data["time"]               ))
        self.time_rel = dict(zip(data["idx"], data["time"] - self.pb_time))

    def get_luminosity(self):
        # the number of column in luminosity.d is not fixed...
        # thus, we store the luminosity data as dictionary
        fn = os.path.join(self.path, "luminosity.d")

        # count the # of column in each row
        num_column = check_output("awk '{print NF}' " + fn, shell = True)
        num_column = [int(row)  for row in num_column.strip().split(b"\n")]
        num_column_3 = num_column.count(3)  # number of row that has only 3 columns

        # read the first 3 columns and last 3 columns separately
        time,  Lnu_e, Lnu_ebar = np.genfromtxt(fn, usecols = [0, 1, 2], unpack = True)
        Lnu_x, Nnu_e, Nnu_ebar = np.genfromtxt(fn, usecols = [3, 4, 5], unpack = True, skip_header = num_column_3)
        # add missing data
        Lnu_x    = np.hstack([ [np.NaN] * num_column_3, Lnu_x    ])
        Nnu_e    = np.hstack([ [np.NaN] * num_column_3, Nnu_e    ])
        Nnu_ebar = np.hstack([ [np.NaN] * num_column_3, Nnu_ebar ])

        data = time, Lnu_e, Lnu_ebar, Lnu_x, Nnu_e, Nnu_ebar

        # store into dictionary
        for key, value in zip(fmt_luminosity.names, data):
            self.luminosity[key] = value


class sphynx_binary():
    """
    Contains functions for loading data in binary files, create yt objects,
    and add derived fields in yt object
    """
    def __init__(self, rundir, runpath):
        self.path = os.path.join(runpath, rundir)

        self.binary    = None
        self.binary_fn = None
        self.ds        = None

        self.verbose = False

    def get_all_binary(self, directory = "data"):
        # return the index of all binary files in directory
        fn_list  = glob(os.path.join(self.path, directory, "*"))
        fn_index = [int(fn.split(".")[-1])  for fn in sorted(fn_list)]
        return fn_index

    def get_binary(self, fn):
        # binary files: initial condition & data/
        fn = os.path.join(self.path, fn)

        if fn != self.binary_fn:
            self.binary    = np.fromfile(fn, dtype = fmt_binary)
            self.binary_fn = fn
            if self.verbose:
                print("File {} is loaded.".format(self.binary_fn), flush = True)

    def create_yt_obj(self, fn, n_ref = 64):
        # create a yt object using data in binary files
        self.get_binary(fn)

        data = {'particle_position_x':      self.binary["x"],
                'particle_position_y':      self.binary["y"],
                'particle_position_z':      self.binary["z"],
                'particle_velocity_x':      self.binary["vx"],
                'particle_velocity_y':      self.binary["vy"],
                'particle_velocity_z':      self.binary["vz"],
                'particle_index':           self.binary["idx"],
                'smoothing_length':         self.binary["h"],
                'particle_gas_density':     self.binary["promro"],
                'particle_gas_temperature': self.binary["temp"],
                'particle_mass':            self.calc_parmass()   }

        var_known = set(["x", "y", "z", "vx", "vy", "vz", "idx", "h", "promro", "temp"])

        for var in quant_binary:
            if var not in var_known:
                data[var] = self.binary[var]

        bbox = np.array( [ (self.binary[idx].min(), self.binary[idx].max())  for idx in "xyz" ] )

        self.ds = yt.load_particles(data, n_ref = n_ref, bbox = bbox,
                                    length_unit = yt.units.cm, mass_unit = yt.units.g, time_unit = yt.units.s)

        # fake command to create the field list?!
        # Otherwise the ds object would not have the attribute field_info
        self.ds.field_list;

        print("File {} is ported into yt object self.ds.".format(self.binary_fn), flush = True)

    def add_yt_smoothed_field(self, quant, kernel = "cubic"):
        # add yt derived, smoothed field
        add_volume_weighted_smoothed_field("all", "particle_position", "particle_mass",
                                           "smoothing_length", "particle_gas_density",
                                           quant, self.ds.field_info, kernel_name = kernel)

    def calc_parmass(self):
        # compute the particle mass
        return np.diff(self.binary["mass_cum"], prepend = 0.)

    def calc_quadmom(self):
        # compute the 2nd-order time derivative of mass quadrupole moment
        parmass = self.calc_parmass()  # particle mass

        x  = self.binary["x"]
        y  = self.binary["y"]
        z  = self.binary["z"]

        vx = self.binary["vx"]
        vy = self.binary["vy"]
        vz = self.binary["vz"]

        # adopt from actualizamod.f90
        ax = -(self.binary["gradp_x"] - const_G * self.binary["fx"])
        ay = -(self.binary["gradp_y"] - const_G * self.binary["fy"])
        az = -(self.binary["gradp_z"] - const_G * self.binary["fz"])

        trace  = vx**2 + vy**2 + vz**2 + (x * ax  + y * ay + z * az)
        trace *= 2. / 3

        Ixx = 2 * (vx**2 + x * ax) - trace
        Iyy = 2 * (vy**2 + y * ay) - trace
        Izz = 2 * (vz**2 + z * az) - trace
        Ixy = 2 * (vx * vy) + (x * ay + y * ax)
        Iyz = 2 * (vy * vz) + (y * az + z * ay)
        Ixz = 2 * (vz * vx) + (z * ax + x * az)

        I_all = Ixx, Ixy, Ixz, Iyy, Iyz, Izz

        return [I * parmass  for I in I_all]

    def calc_omega(self):
        # compute the angular velocity
        x  = self.binary["x"]
        y  = self.binary["y"]

        vx = self.binary["vx"]
        vy = self.binary["vy"]

        return (vx * y - vy * x) / (x * x + y * y)


class sphynx_par():
    """
    Read the simulation setting in parameters.f90 and parameters_idsa.f90
    """
    _pattern = re.compile(r"(PRECISION|LOGICAL|INTEGER|CHARACTER).*PARAMETER::(\S*\s*=\s*\S*)")

    def __init__(self, rundir, runpath):
        self.path = os.path.join(runpath, rundir)
        self.par  = dict()

        self._load_parameter(os.path.join(self.path, "parameters.f90"))
        self._load_parameter(os.path.join(self.path, "parameters_idsa.f90"))

    def _load_parameter(self, fn):
        """
        Use the regular pattern '_pattern' to obtain all parameters defined in 'fn'
        """
        with open(fn) as f:
            for line in f:
                line = line.strip()

                # skip empty lines and comments
                if not line or line[0] == "!":
                    continue

                # use regular pattern to retrieve info
                data = self._pattern.search(line)

                if data is None:
                    continue

                datatype, expr = data.groups()

                # remove trailing ! and following comment
                if '!' in expr:
                    expr = expr.split("!")[0]

                # replace 'd' to 'e'
                if datatype == "PRECISION":
                    expr = expr.replace("d", "e")

                # replace true/false
                if datatype == "LOGICAL":
                    expr = expr.replace(".false.", "False").replace(".true.", "True")

                varname = expr.split("=")[0]
                varname = varname.strip()
                try:
                    exec(expr)
                except SyntaxError:
                    print("Unknown parameter setting in {}: {}".format(fn, expr), flush = True)
                    continue

                # store to self.par
                self.par[varname.lower()] = eval(varname)  # use keyword in lower case

    def getpar(self, key):
        """
        Return value stored in 'par'. The 'key' can be either upper or lower case.

        If no data found, return None
        """
        return self.par.get(key.lower(), None)


class sphynx(sphynx_ascii, sphynx_binary, sphynx_par):
    """
    Contains functions for visualization
    """
    gw_text = {"plus" : r"$h_\plus$",
               "cross": r"$h_\mathrm{x}$",
               "both" : r"$h$"             }

    def __init__(self, rundir, runpath, load_eos = False):
        sphynx_ascii.__init__(self, rundir = rundir, runpath = runpath)
        sphynx_binary.__init__(self, rundir = rundir, runpath = runpath)
        sphynx_par.__init__(self, rundir = rundir, runpath = runpath)
        self.path = os.path.join(runpath, rundir)

        self.get_bounce()
        self.get_tiempos()

        # Since NuclearEOS can be instantized once only, it would be better
        # to create the NuclearEOS object outside the sphynx object
        self.fn_eos = os.path.join(self.path, self.getpar("eostab"))
        self.neos   = None

        if load_eos:
            self._load_eostable()

    def __str__(self):
        return self.path

    def _load_eostable(self):
        self.fn_eos = os.path.join(self.path, self.getpar("eostab"))
        self.neos   = NuclearEOS(self.fn_eos)

    def _free_eostable(self):
        if self.neos is not None:
            self.neos.del_table()
            self.neos = None

    def _check_iterable(self, obj, index = None):
        # check whether obj (or obj[index] if index is set) is iterable
        # if not, convert obj to tuple
        test_obj = obj[index]  if index  else  obj

        if not isinstance(test_obj, Iterable):
            obj = obj,

        return obj

    def _get_evolution(self, quant, tscale = "pb"):
        """
        Get the evolution of given quantity 'quant'. Currently, only support quantities in

            central_values.d
            luminosity.d
            radgrav.d
        """
        if quant in quant_central:
            key = "central_values"
        elif quant in quant_lumin:
            key = "luminosity"
        elif quant in quant_radgrav:
            key = "radgrav"
        else:
            raise ValueError("Unknown quantity: {}".format(quant))

        if not self._get_ascii_table(key):
            self.ascii_loader[key]()

        table = self._get_ascii_table(key)
        time  = table["t"]
        data  = table[quant]

        if tscale == "pb":
            time = time - self.pb_time

        return time, data

    def get_profile(self, nout):
        # get the data in the binary file in directory 'data'
        fn = os.path.join("data", "s1.{:06d}".format(nout))
        self.get_binary(fn)

    def get_initmodel(self):
        # get the data in the initial model 'archivo'
        fn = self.getpar('archivo')
        fn = os.path.join("..", "initialmodels", fn)  # path relative to the run directory
        self.get_binary(fn)

    def get_ytobj(self, nout, n_ref = 64):
        # create the yt object from the binary file in directory 'data'
        fn = os.path.join("data", "s1.{:06d}".format(nout))
        self.create_yt_obj(fn, n_ref)

    def _get_xlabel(self, tscale = "pb"):
        # choose the x label
        if self.pb and tscale == "pb":
            return r"$t_\mathrm{pb}$ [ms]"
        else:
            return r"Time [ms]"

    def _get_gw_text(self, mode):
        return self.gw_text[mode]

    def _get_gw_ylabel(self, mode):
        # choose the y label for gw plot
        return r"$10^{21}$" + self._get_gw_text(mode)

    def _create_figure(self, ax):
        if ax is None:
            return plt.subplots()  # create a new figure and axes object
        else:
            return None, ax        # do noting

    def calc_gw_strain(self, phi, theta, dist = 10, mode = "both"):
        """
        Compute the GW strains

        Parameters
        ----------
        phi:   azimuthal angle in radian
        theta: polar angle in radian
        dist:  distance in kpc
        mode:  "both"  -> include both A_plus and A_cross
               "plus"  -> A_plus only
               "cross" -> A_cross only
        """
        assert mode in ["plus", "cross", "both"], "Unknown mode: {}".format(mode)

        if self.radgrav is None:
            self.get_radgrav()

        # call functions in lib_gw to compute GW strain
        Ipp, Itp, Itt = conv_cart2sph(self.radgrav["Ixx"],
                                      self.radgrav["Ixy"],
                                      self.radgrav["Ixz"],
                                      self.radgrav["Iyy"],
                                      self.radgrav["Iyz"],
                                      self.radgrav["Izz"],
                                      phi = phi, theta = theta)

        A_plus, A_cross = calc_amplitude(Ipp, Itp, Itt)

        # determine which mode to be returned
        if mode == "plus":
            gw_strain = A_plus
        elif mode == "cross":
            gw_strain = A_cross
        else:
            gw_strain = A_plus + A_cross

        # take care the unit and distance
        gw_strain *= const_G / const_c**4 / (dist * kpc2cm)

        return gw_strain

    def calc_separated_strain(self, nouts, rlimit = None, fnout = None):
        """
        Compute the GW strain in different regions, and output it to an ASCII file

        Parameters
        ----------
        nouts: sequence of noutput to be output
        rlimit: if None, compute the total GW strain
                if rlimit = [r_1, r_2, ..., r_n], compute the GW strain in regions
                [0, r_1], [r_i, r_i+1], ..., [r_n, infinity], where all the values are in km
        fnout: string. The filename of the ASCII output
        """
        nouts = self._check_iterable(nouts)

        if fnout is None:
            fnout = "radgrav_separated.d"

        fnout = os.path.join(self.path, fnout)

        with open(fnout, "w") as f:
            # prepare the header
            header   = ["#", "N_output", "Time(second)"]
            header_I = ["Ixx", "Ixy", "Ixz", "Iyy", "Iyz", "Izz"]

            if rlimit is None:
                header += header_I
            else:
                header += [i + "(r < {} km)".format(rlimit[0])  for i in header_I]

                for idx, rout in enumerate(rlimit[1:], 1):
                    rin = rlimit[idx - 1]
                    header += [i + "({} < r < {} km)".format(rin, rout)  for i in header_I]

                header += [i + "(r > {} km)".format(rlimit[-1])  for i in header_I]

            header = "    ".join(header)
            f.write(header + "\n")

            # compute the separated strain here
            def sum_region(radii, data, r1 = None, r2 = None):
                # function that sums the values in specified region [r1, r2]
                if r2 is None:
                    cond = radii < r1
                elif r1 is None:
                    cond = radii >= r2
                else:
                    cond = (r1 <= radii) & (radii < r2)

                return np.sum(data[cond])

            for nout in nouts:
                self.get_profile(nout)
                radius = self.binary["radius"] / 1e5  # to km
                I_all  = self.calc_quadmom()

                # prepare the data
                data_prefix = "{:6d}  {:.7e}  ".format(nout, self.time[nout])

                data = list()
                if rlimit is None:
                    data += [np.sum(I)  for I in I_all]
                else:
                    data += [sum_region(radius, I, r1 = rlimit[0])  for I in I_all]

                    for idx, r2 in enumerate(rlimit[1:], 1):
                        r1 = rlimit[idx - 1]
                        data += [sum_region(radius, I, r1 = r1, r2 = r2)  for I in I_all]

                    data += [sum_region(radius, I, r2 = rlimit[-1])  for I in I_all]

                data = ["{:.7e}".format(i)  for i in data]

                f.write(data_prefix + "  ".join(data) + "\n")

        print("Output file: {}".format(fnout), flush = True)

    def calc_BV_frequency(self, nout, dr = 1e5, rmax = 2e7, rsh = None,
                          method = "binary", bin_data = False, neos = None, **kwargs):
        """
        Parameters
        ----------
        nout: scalar or sequence of output indices
        dr: the bin width and the center of lowest bin
        rmax: the center of highest bin
        rsh: shock radius. Set to rmax if not specified
        method: if method == "yt"    : using binned yt-redndering meshgrid data
                             "binary": using binned SPHYNX's binary output
                             "sph"   : using SPHYNX's binary output and SPH formalism for gradient
        bin_data: Boolean. Only works if method == "sph"
                  If True, the bv from "sph" method is binned
        """
        assert method in ["binary", "yt", "sph"], "Unknown method: {}".format(method)

        if neos is None:
            if self.neos is None:
                raise ValueError ("No nuclear EoS table specified! "
                                  "Run self._load_eostable() to load table "
                                  "or assign a NuclearEOS instance to 'neos'.")
            else:
                neos = self.neos

        if isinstance(nout, Iterable): # case: multiple nout specified
#            BV_freq = dict()
            BV_freq = [None] * len(nout)

            for idx, n in enumerate(nout):
                radius, bv = self.calc_BV_frequency(n, dr = dr, rmax = rmax, rsh = rsh,
                                                    method = method, bin_data = bin_data, neos = neos)
                BV_freq[idx] = bv

            return radius, np.array(BV_freq)

        else: # case: one nout specified
            BV_obj  = BV_frequency(neos)

            if method == "binary":
                self.get_profile(nout)
                radius, bv = BV_obj.get_bv_sphynx_bin(self.binary, dr = dr, rmax = rmax, rsh = rsh)
            elif method == "sph":
                self.get_profile(nout)
                radius, bv = BV_obj.get_bv_sphynx_sph(self.binary, dr = dr, rmax = rmax, rsh = rsh,
                                                      kernel_idx = self.getpar("kernel"), bin_data = bin_data)
            else:
                self.get_ytobj(nout)

                ### create needed derived fields
                # radius of each meshgrid
                def _radius(field, data):
                    return data[('index', 'radius')]

                self.ds.add_field(("deposit", "mesh_radius"),
                                  function = _radius, units = "cm", sampling_type = "cell")

                # smoothed field
                for quant in ["particle_gas_density", "s", "ye"]:
                    self.add_yt_smoothed_field(quant)

                def _mass(field, data):
                    return data[('deposit', 'all_smoothed_particle_gas_density')] * data[('gas', 'cell_volume')]

                self.ds.add_field(("deposit", "all_smoothed_mass"),
                                  function = _mass, units = "code_mass", sampling_type = "cell")

                radius, bv = BV_obj.get_bv_yt(self.ds, dr = dr, rmax = rmax, rsh = rsh)

            return radius, bv

    def calc_SH_criterion(self, nout, dr = 1e5, rmax = 2e7, rsh = None,
                          method = "binary", bin_data = False, neos = None, **kwargs):
        """
        Parameters
        ----------
        nout: scalar or sequence of output indices
        dr: the bin width and the center of lowest bin
        rmax: the center of highest bin
        rsh: shock radius. Set to rmax if not specified
        method: if method == "yt"    : using binned yt-redndering meshgrid data
                             "binary": using binned SPHYNX's binary output
                             "sph"   : using SPHYNX's binary output and SPH formalism for gradient
        bin_data: Boolean. Only works if method == "sph"
                  If True, the R_SH from "sph" method is binned
        """
        assert method in ["binary", "yt", "sph"], "Unknown method: {}".format(method)

        if neos is None:
            if self.neos is None:
                raise ValueError ("No nuclear EoS table specified! "
                                  "Run self._load_eostable() to load table "
                                  "or assign a NuclearEOS instance to 'neos'.")
            else:
                neos = self.neos

        if isinstance(nout, Iterable): # case: multiple nout specified
            SH_crit = [None] * len(nout)

            for idx, n in enumerate(nout):
                radius, sh = self.calc_SH_criterion(n, dr = dr, rmax = rmax, rsh = rsh,
                                                    method = method, bin_data = bin_data, neos = neos)
                SH_crit[idx] = sh

            return radius, np.array(SH_crit)

        else: # case: one nout specified
            SH_obj  = SH_criterion(neos)

            if method == "binary":
                self.get_profile(nout)
                self.binary["dummy1"] = self.calc_omega()  # compute omega and store into dummy1
                radius, sh = SH_obj.get_sh_sphynx_bin(self.binary, dr = dr, rmax = rmax, rsh = rsh)
            elif method == "sph":
                self.get_profile(nout)
                self.binary["dummy1"] = self.calc_omega()  # compute omega and store into dummy1
                radius, sh = SH_obj.get_sh_sphynx_sph(self.binary, dr = dr, rmax = rmax, rsh = rsh,
                                                      kernel_idx = self.getpar("kernel"), bin_data = bin_data)
            else:
                self.get_ytobj(nout)

                ### create needed derived fields
                # radius of each meshgrid
                def _radius(field, data):
                    return data[('index', 'radius')]

                self.ds.add_field(("deposit", "mesh_radius"),
                                  function = _radius, units = "cm", sampling_type = "cell")

                # omega: angular velocity about the z direction
                def _omega(field, data):
                    x  = ("all", "particle_position_x")
                    y  = ("all", "particle_position_y")
                    vx = ("all", "particle_velocity_x")
                    vy = ("all", "particle_velocity_y")

                    return (data[vx] * data[y] - data[vy] * data[x]) / (data[x]**2 + data[y]**2)

                self.ds.add_field(("all", "omega_phi"),
                                  function = _omega, units = "cm/(code_length*s)", particle_type = True)

                # smoothed field
                for quant in ["particle_gas_density", "s", "ye", "omega_phi"]:
                    self.add_yt_smoothed_field(quant)

                def _mass(field, data):
                    return data[('deposit', 'all_smoothed_particle_gas_density')] * data[('gas', 'cell_volume')]

                self.ds.add_field(("deposit", "all_smoothed_mass"),
                                  function = _mass, units = "code_mass", sampling_type = "cell")

                radius, sh = SH_obj.get_sh_yt(self.ds, dr = dr, rmax = rmax, rsh = rsh)

            return radius, sh

    def calc_vaniso(self, nout, dr = 1e5, rmax = 2e7, rsh = None, method = "binary", **kwargs):
        """
        Parameters
        ----------
        nout: scalar or sequence of output indices
        dr: the bin width and the center of lowest bin
        rmax: the center of highest bin
        rsh: shock radius. Set to rmax if not specified
        method: if method == "binary": using binned SPHYNX's binary output
        """
        assert method in ["binary"], "Unknown method: {}".format(method)

        if isinstance(nout, Iterable): # case: multiple nout specified
            vaniso = [None] * len(nout)

            for idx, n in enumerate(nout):
                radius, v_std = self.calc_vaniso(n, dr = dr, rmax = rmax, rsh = rsh, method = method)
                vaniso[idx] = v_std

            return radius, np.array(vaniso)

        else: # case: one nout specified
            Vaniso_obj  = Vaniso()

            if method == "binary":
                self.get_profile(nout)
                radius, v_std = Vaniso_obj.get_vaniso_sphynx_bin(self.binary, dr = dr, rmax = rmax, rsh = rsh)

            return radius, v_std

    def calc_sphharm(self, nout, dens, ls, ms, fraction = 0.01, method = "binary", **kwargs):
        """
        Parameters
        ----------
        nout: scalar or sequence of output indices
        dens, fraction: consider the shell with density: [dens * (1 - fraction), dens * (1 + fraction)]
        ls, ms: sequences of the spherical harmonic mode (l, m)
        method: if method == "binary": using binned SPHYNX's binary output
        """
        assert method in ["binary"], "Unknown method: {}".format(method)

        if isinstance(nout, Iterable): # case: multiple nout specified
            coe = [None] * len(nout)

            for idx, n in enumerate(nout):
                coe_dict = self.calc_sphharm(n, dens, ls, ms, fraction = fraction, method = method)
                coe[idx] = coe_dict

            # combine all dictionaries into one
            coe_all = dict()

            for lm in zip(ls, ms):
                coe_all[lm] = np.array( [dict_obj.get(lm, "None")  for dict_obj in coe] )

            return coe_all

        else: # case: one nout specified
            sphharm_obj = SphHarm_decomp()

            if method == "binary":
                self.get_profile(nout)
                coe_dict = sphharm_obj.get_sphharm_decomp_sphynx(self.binary, dens, ls, ms, fraction = fraction)

            return coe_dict

    def plot_evolution(self, quant, tscale = "pb", ax = None, labels = None, **kwargs):
        """
        Plot the evolution of quantity.

        Parameters
        ----------
        tscale: "pb": show the time scale relative to t_pb
                Otherwise: show the physical time
        """
        quant = self._check_iterable(quant)

        if labels is None:
            labels = quant
        else:
            labels = self._check_iterable(labels)

        # load data
        data_all = [self._get_evolution(q, tscale)  for q in quant]

        # plot
        fig, ax = self._create_figure(ax)

        for (time, data), label in zip(data_all, labels):
            ax.plot(time * sec2ms, data, label = label, **kwargs)

        label_x = self._get_xlabel(tscale)
        ax.set_xlabel(label_x)
        ax.set_ylabel(quant)

        if len(quant) > 1:
            ax.legend(framealpha = 0)

        if fig is not None:
            return fig, ax

    def plot_profile(self, x, y, xfac = 1., yfac = 1.,
                     ax = None, label = None, **kwargs):
        """
        Plot the profile.

        Parameters
        ----------
        x: String. keyword in the binary file for the x-axis
        y: String. keyword in the binary file for the y-axis
        xfac: Numeric. Factor of the x-axis
        yfac: Numeric. Factor of the y-axis
        """
        assert self.binary is not None, "No binary file loaded."
        assert x in quant_binary,       "Unknown variable {}.".format(x)
        assert y in quant_binary,       "Unknown variable {}.".format(y)

        # plot
        fig, ax = self._create_figure(ax)

        ax.plot(xfac * self.binary[x], yfac * self.binary[y], label = label, **kwargs)

        if xfac != 1.0:
            x = r"{:.1e} $\times$ {}".format(xfac, x)
        if yfac != 1.0:
            y = r"{:.1e} $\times$ {}".format(yfac, y)

        ax.set_xlabel(x)
        ax.set_ylabel(y)

        if fig is not None:
            return fig, ax

    def plot_gw_strain(self, phi, theta, dist = 10, mode = "both", ax = None, **kwargs):
        """
        Plot the GW strains
        """
        gw_strain = self.calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)

        # plot
        fig, ax = self._create_figure(ax)

        ax.plot((self.radgrav["t"] - self.pb_time) * sec2ms, gw_strain * 1e21, **kwargs)

        label_x = self._get_xlabel()
        label_y = self._get_gw_ylabel(mode)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(r"$\phi = {:.2f} \pi, \theta = {:.2f} \pi$".format(phi / np.pi, theta / np.pi))

        if fig is not None:
            return fig, ax

    def plot_gw_spectrum(self, phi, theta, dist = 10, mode = "both", method = "periodogram",
                         dt = None, tstart = None, tend = None, ax = None, **kwargs):
        """
        Plot the spectra of GW strains
        """
        assert method in ["kcpan", "moore14", "periodogram"], "Unknown method: {}.".format(method)

        gw_strain = self.calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)
        gw_time   = self.radgrav["t"] - self.pb_time

        if method == "periodogram":
            freq, hchar  = calc_spectrum_periodogram(gw_time, gw_strain,
                                                     tstart = tstart, tend = tend, dt = dt)
        else:
            freq, hchar  = calc_spectrum_fft(gw_time, gw_strain,
                                             method = method, dist = dist,
                                             tstart = tstart, tend = tend, dt = dt)

        # plot
        fig, ax  = self._create_figure(ax)
        label_y  = self._get_gw_text(mode)
        label_y += "$_{,\mathrm{char}}$"  if "_" in label_y  else "$_{\mathrm{char}}$"

        ax.loglog(freq, hchar, **kwargs)
        ax.set_xlabel(r"$f$ [Hz]")
        ax.set_ylabel(label_y)
        ax.set_title(r"$\phi = {:.2f} \pi, \theta = {:.2f} \pi$".format(phi / np.pi, theta / np.pi))

        if fig is not None:
            return fig, ax

    def plot_gw_spectrogram(self, phi, theta, dist = 10, mode = "both", method = "stft",
                            window = None, tstart = None, tend = None,
                            numpt = 1000, dt = 1e-5,
                            ax = None, fmax = 4000, clim = [1e-24, 3e-22], **kwargs):
        """
        Plot the spectrogram of GW strain

        Parameters
        ----------
        see the docstring of plot_gw
        """
        assert method in ["fft", "stft"], "Unknown method: {}.".format(method)

        gw_strain = self.calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)
        gw_time   = self.radgrav["t"] - self.pb_time

        if method == "fft":
            times, freq, spectrogram = calc_spectrogram_fft(gw_time, gw_strain,
                                                            window = window, tstart = tstart, tend = tend,
                                                            numpt = numpt)
        else:
            times, freq, spectrogram = calc_spectrogram_stft(gw_time, gw_strain,
                                                             window = window, tstart = tstart, tend = tend,
                                                             dt = dt)

        # plot
        fig, ax = self._create_figure(ax)

        if method == "fft":
            img = ax.imshow(spectrogram.T,
                            extent = [times[0] * sec2ms, times[-1] * sec2ms, freq[0], freq[-1]],
                            interpolation = "gaussian",
                            aspect = 0.12 * times[-1] * (4000.0 / fmax),
                            vmin = clim[0],
                            vmax = clim[1],
                            cmap = "viridis",
                            norm = mpl.colors.LogNorm(),
                            origin ='lower')
        else:
            img = ax.pcolormesh(times * sec2ms, freq, spectrogram,
                                vmin = clim[0],
                                vmax = clim[1],
                                shading = "gouraud")

        # TODO: adjust colorbar
        cbar = plt.colorbar(img, ax = ax)

        label_x = self._get_xlabel()
        ax.set_xlabel(label_x)
        ax.set_ylabel("Frequency [Hz]")
        ax.set_title(r"$\phi = {:.2f} \pi, \theta = {:.2f} \pi$".format(phi / np.pi, theta / np.pi))
        ax.set_ylim([0, fmax])

        if fig is not None:
            return fig, ax

    def plot_gw(self, phi, theta, dist = 10, mode = "both", method = "stft",
                window = None, tstart = None, tend = None,
                numpt = 1000, dt = 1e-5,
                fmax = 4000, clim = [1e-24, 3e-22], aspect_cb = 16):
        """
        Plot the GW strain and spectrogram at the given angle 'phi' and 'theta'.

        Adopted from Kuo-Chuan's code.

        Parameters
        ----------
        phi:   azimuthal angle in radian
        theta: polar angle in radian
        dist:  distance in kpc
        mode:  "both"  -> include both A_plus and A_cross
               "plus"  -> A_plus only
               "cross" -> A_cross only
        window: width of window function in ms
        tstart: the start time of the spectrogram in sec (relative to t_pb)
        tend:   the final time of the spectrogram in sec (relative to t_pb)
        numpt:  number of points in the time domain [t - window, t + window]
        fmax:   the upper limit of frequency displayed in the spectrogram plot, in Hz
        clim:   the lower and upper limit of GW strain displayed in the spectrogram plot
        """
        assert method in ["fft", "stft"], "Unknown method: {}.".format(method)

        gw_strain = self.calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)
        gw_time   = self.radgrav["t"] - self.pb_time

        if method == "fft":
            times, freq, spectrogram = calc_spectrogram_fft(gw_time, gw_strain,
                                                            window = window, tstart = tstart, tend = tend,
                                                            numpt = numpt)
        else:
            times, freq, spectrogram = calc_spectrogram_stft(gw_time, gw_strain,
                                                             window = window, tstart = tstart, tend = tend,
                                                             dt = dt)

        # plto here
        fig = plt.figure(figsize = (10, 8))

        gs = gridspec.GridSpec(2, 2, width_ratios = [1, 0.03], height_ratios = [1, 4])
        gs.update(left = 0.15, right = 0.9, hspace = 0.05, wspace = 0.02)

        # Amplitude
        ax1 = plt.subplot(gs[0])
        ax1.plot(gw_time * sec2ms, gw_strain * 1e21, linewidth = 1)

        ylim = np.max(np.abs(gw_strain * 1e21))
        ylim = np.int(ylim) + 1

        ax1.set_xlim([times[0] * sec2ms, times[-1] * sec2ms])
        ax1.set_ylim([-ylim, ylim])
        ax1.set_yticks([-ylim, 0, ylim])

        label_y = self._get_gw_ylabel(mode)
        ax1.set_ylabel(label_y, labelpad = 10, size = 24)
        ax1.set_title(r"$\phi = {:.2f} \pi, \theta = {:.2f} \pi$".format(phi / np.pi, theta / np.pi),
                      size = 24, y = 1.2)

        # Spectrogram
        ax2 = plt.subplot(gs[2])

        if method == "fft":
            im = ax2.imshow(spectrogram.T,
                            extent = [times[0] * sec2ms, times[-1] * sec2ms, freq[0], freq[-1]],
                            interpolation = "gaussian",
                            aspect = 0.12 * times[-1] * (4000.0 / fmax),
                            vmin = clim[0],
                            vmax = clim[1],
                            cmap = "viridis",
                            norm = mpl.colors.LogNorm(),
                            origin ='lower')
        else:
            im = ax2.pcolormesh(times * sec2ms, freq, spectrogram,
                                vmin = clim[0],
                                vmax = clim[1],
                                shading = "gouraud")

        label_x = self._get_xlabel()
        text    = self._get_gw_text(mode)
        xdomain = ax2.get_xlim()

        ax2.text(xdomain[0] + 0.05 * np.diff(xdomain), (0.9 * fmax), text, fontsize = 24, color = "w")
        ax2.set_xlabel(label_x, size = 24)
        ax2.set_ylabel("Frequency [Hz]", labelpad = 20, size = 24)
        ax2.set_ylim([0, fmax])

        # colorbar
        ax3 = plt.subplot(gs[3])
        ax3.set_aspect(aspect_cb)

        cbar = plt.colorbar(im, cax = ax3, use_gridspec = True)
        cbar.mappable.set_clim(clim)

        # align ylabel of ax1 and ax2
        fig.align_ylabels([ax1, ax2])
        plt.setp(ax1.get_xticklabels(), visible = False)

        for ax in [ax1, ax2, ax3]:
            plt.setp(ax.get_yticklabels(), fontsize = 20)

        plt.setp(ax1.get_xticklabels(), fontsize = 20)

        return fig, gs

    def plot_profile_evol(self, quant, tend = None, dr = 1e5, rmax = 2e7, rsh = None,
                          method = "binary", bin_data = False, neos = None,
                          include_vphi = False, include_vphi_ave = False,
                          include_vtheta = False, include_vtheta_ave = False,
                          ax = None, vminmax = None, **kwargs):
        """
        Plot the evolution of derived profile after core bounce in contour plot
        in certain range (t_pb, tend), where tend = min(tend, last time).

        TODO:
            for method = "sph", the radius is not fixed and thus incorrect

        Parameters
        ----------
        quant: quantity to be plotted
               Support quantity: Brunt-Vaisala frequency   (quant = "bv")
                                 Solberg-Hoiland criterion (quant = "sh")
        vminmax: the vmin and vmax, where vmin = -vminmax and vmax = vminmax
        """
        assert quant in ["bv", "sh", "vaniso"], "Unknown quantity: {}".format(quant)

        nouts = [nout  for nout in self.get_all_binary()  if nout >= self.pb_nout]
        times = [self.time_rel[i] * sec2ms  for i in nouts]  # in ms

        # trim data if tend is specified
        if tend is not None:
            idx_trim = bisect(times, tend)
            nouts    = nouts[:idx_trim]
            times    = times[:idx_trim]

        # obtain the data and setting cmap
        if quant == "bv":
            func = self.calc_BV_frequency
            cmap = "bwr"
        elif quant == "sh":
            func = self.calc_SH_criterion
            cmap = "bwr_r"
        elif quant == "vaniso":
            func = self.calc_vaniso
            cmap = "rainbow"

        radial, data = func(nouts, dr = dr, rmax = rmax, rsh = rsh,
                            method = method, bin_data = bin_data, neos = neos,
                            include_vphi = include_vphi, include_vphi_ave = include_vphi_ave,
                            include_vtheta = include_vtheta, include_vtheta_ave = include_vtheta_ave)

        # plot here
        fig, ax = self._create_figure(ax)

        times, radial = np.meshgrid(times, radial)

        if vminmax:
            vmin, vmax = -vminmax, vminmax
        else:
            vmin, vmax = data.min(), data.max()

        if quant == "vaniso":
            norm = None
        else:
            norm = DivergingNorm(vmin = vmin, vcenter = 0, vmax = vmax)

        im = ax.pcolormesh(times, radial / 1e5, data.T,
                           shading = "gouraud", cmap = cmap,
                           vmin = vmin, vmax = vmax, norm = norm, **kwargs)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Radius [km]")

        if quant == "bv":
            cb_label = r"$\omega_\mathrm{BV}$ [ms$^{-1}$]"
        elif quant == "sh":
            cb_label = r"$R_\mathrm{SH}$ [ms$^{-2}$]"
        elif quant == "vaniso":
            cb_label = r"$v_\mathrm{aniso}$ [$c$]"

        if vminmax:
            cbar = plt.colorbar(im, ax = ax, pad = 0.02, extend = "both")
        else:
            cbar = plt.colorbar(im, ax = ax, pad = 0.02)

        cbar.set_label(cb_label, rotation = 270, labelpad = 20)

        if fig is not None:
            return fig, ax

    def plot_sphharm(self, dens, ls = None, ms = None, lms = None, fraction = 0.01, tend = None,
                     ax = None, **kwargs):
        """
        Decompose the shell with density: [dens * (1 - fraction), dens * (1 + fraction)]
        into spherical harmonic components.

        Then, plot the evolution of the coefficient of specified modes
        in certain range (t_pb, tend), where tend = min(tend, last time)

        Parameters
        ----------
        dens, fraction: the density range to be considered
        ls, ms: sequence of the spherical harmonic modes: l and m
        lms: alternative way to assign ls and ms: ls, ms = zip(*lms)
             Override ls and ms if specified.
        """
        assert dens * (1 - fraction) > 0.0, "Negative denisty range is specified."
        assert (ls and ms) or (lms), "ls and ms, or lms, must be set."

        nouts = [nout  for nout in self.get_all_binary()  if nout >= self.pb_nout]
        times = [self.time_rel[i] * sec2ms  for i in nouts]  # in ms

        # trim data if tend is specified
        if tend is not None:
            idx_trim = bisect(times, tend)
            nouts    = nouts[:idx_trim]
            times    = times[:idx_trim]

        # override ls and ms if lms is assigned
        if lms is not None:
            lms = self._check_iterable(lms, index = 0)
        else:
            ls = self._check_iterable(ls)
            ms = self._check_iterable(ms)

            lms = tuple(zip(ls, ms))

        # always calculate a_{0, 0} for normalization
        if (0, 0) not in lms:
            lms = *lms, (0, 0)

        ls, ms = zip(*lms)

        # obtain the data
        data = self.calc_sphharm(nouts, dens, ls, ms, fraction = fraction, method = "binary")

        for lm in lms:
            data[lm] /= data[(0, 0)]  # normalized by a_{0, 0}

        # plot here
        fig, ax = self._create_figure(ax)

        for lm in lms:
            if lm == (0, 0):  # ignore a_{0, 0} case
                continue

            label = r"$a_{" + "{}{}".format(*lm) + r"}$"
            ax.plot(times, data[lm], label = label, **kwargs)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel(r"$a_{lm} / a_{00}$")
        ax.set_title("Density: {:.2e} [gram]".format(dens))
        ax.legend()

        if fig is not None:
            return fig, ax

    def plot_sphharm_amplitude(self, dens, l, fraction = 0.01, tend = None,
                               ax = None, **kwargs):
        """
        Similar to plot_sphharm(), but plot the square root of the sum of the squares
        of the a_{l, m}, sqrt( P(l) ), for the given l and density.

            P(l) = sum_{m = -l, ..., l} a_{l, m}^2

        Parameters
        ----------
        dens, fraction: the density range to be considered
        l: the spherical harmonic modes: l
        """
        assert l > 0, "Invalid l: {}".format(l)  # l = 0 is meaningless

        dens = self._check_iterable(dens)
        dens = sorted(dens)

        for d in dens:
            assert d * (1 - fraction) > 0.0, "Negative denisty range is specified."

        nouts = [nout  for nout in self.get_all_binary()  if nout >= self.pb_nout]
        times = [self.time_rel[i] * sec2ms  for i in nouts]  # in ms

        # trim data if tend is specified
        if tend is not None:
            idx_trim = bisect(times, tend)
            nouts    = nouts[:idx_trim]
            times    = times[:idx_trim]

        lms = [(l, m)  for m in range(-l, l + 1)] + [(0, 0)]
        ls, ms = zip(*lms)

        # obtain the data
        data_all = list()

        for d in dens:
            coe_all = self.calc_sphharm(nouts, d, ls, ms, fraction = fraction, method = "binary")

            # compute P(l) and sqrt( P(l) )
            P_l = np.sum( [coe * coe  for lm, coe in coe_all.items()  if lm[0] == l], axis = 0 )
            a_l = np.sqrt(P_l) / coe_all[(0, 0)]  # normalized by a_0

            data_all.append(a_l)

        # plot here
        fig, ax = self._create_figure(ax)

        for d, data in zip(dens, data_all):
            label = r"$\rho = $" + "{:.1e}".format(d)
            ax.plot(times, data, label = label, **kwargs)

        ax.set_xlabel("Time [ms]")
        ax.set_ylabel("$a_{}".format(l) + "/ a_{0}$")
        ax.legend()

        if fig is not None:
            return fig, ax

