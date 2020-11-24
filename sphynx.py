#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Class objects that contains functions for visualizing SPHYNX resutls
#
#  Last Updated: 2020/11/03
#  He-Feng Hsieh


import os
import re
import yt
import numpy as np
import matplotlib as mpl
from glob import glob
from matplotlib import pyplot as plt
from matplotlib import gridspec as gridspec
from .fmt import *
from .lib_gw import *


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


class sphynx_loader():
    """
    Contains functions for loading SPHYNX's ASCII results
    """
    def __init__(self, rundir, runpath):
        self.path = os.path.join(runpath, rundir)

        self.central_values = None
        self.radgrav        = None
        self.luminosity     = None
        self.pb             = None
        self.pb_time        = None
        self.pb_nout        = None
        self.time           = None
        self.time_rel       = None  # time relative to bounce

    def _load_central_values(self):
        # data in central_values.d
        fn = os.path.join(self.path, "central_values.d")
#        return np.genfromtxt(fn, fmt_central_values)
        self.central_values = np.genfromtxt(fn, fmt_central_values)

    def _load_radgrav(self):
        # data in central_values.d
        fn = os.path.join(self.path, "radgrav.d")
        self.radgrav = np.genfromtxt(fn, fmt_radgrav)

    def _load_luminosity(self):
        # data in central_values.d
        fn = os.path.join(self.path, "luminosity.d")
        self.luminosity = np.genfromtxt(fn, fmt_luminosity)

    def _load_tiempos(self):
        # data in tiempos.d
        fn   = os.path.join(self.path, "tiempos.d")
        data = np.genfromtxt(fn, fmt_tiempos)

        self.time     = dict(zip(data["idx"], data["time"] ))
        # if the simulation restarts from postbounce, the relative time is not correct!
        # here we obtain the relative time from the physical time
        self.time_rel = dict(zip(data["idx"], data["time"] - self.pb_time))
#        self.time_rel = dict(zip(data["idx"], data["time_rel"]))

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
        self._load_central_values()

    def get_radgrav(self):
        self._load_radgrav()

    def get_luminosity(self):
        self._load_luminosity()

    def get_tiempos(self):
        self._load_tiempos()


class sphynx_binary():
    """
    Contains functions for loading data in binary files
    """
    def __init__(self, rundir, runpath):
        self.path = os.path.join(runpath, rundir)

        self.binary    = None
        self.binary_fn = None
        self.ds        = None

    def _load_binary(self, fn):
        # binary files: initial condition & data/
        fn = os.path.join(self.path, fn)
        self.binary    = np.fromfile(fn, fmt_binary)
        self.binary_fn = fn

    def get_all_binary(self, directory = "data"):
        # return the index of all binary files in directory
        fn_list = glob(os.path.join(self.path, directory, "*"))
        fn_index = [int(fn.split(".")[-1])  for fn in sorted(fn_list)]
        return fn_index

    def get_binary(self, fn):
        self._load_binary(fn)
        print("File {} is loaded.".format(self.binary_fn))

    def create_ytobj(self, fn, n_ref = 64):
        # create a yt object using data in binary files
        self._load_binary(fn)

        # compute particle mass
        par_mass = np.diff(self.binary["mass_cum"])[0]
        par_mass = np.ones(self.binary["idx"].size) * par_mass

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
                'particle_mass':            par_mass               }

        var_known = set(["x", "y", "z", "vx", "vy", "vz", "idx", "h", "promro", "temp"])

        for var in quant_binary:
            if var not in var_known:
                data[var] = self.binary[var]

        bbox = np.array([(self.binary[idx].min(), self.binary[idx].max())
                         for idx in "xyz"])

        self.ds = yt.load_particles(data, n_ref = n_ref, bbox = bbox,
                                    length_unit = yt.units.cm, mass_unit = yt.units.g, time_unit = yt.units.s)

        # create the field list?
        # Otherwise the ds object would not have the attribute field_info
        self.ds.field_list;

        print("File {} is ported into yt object self.ds.".format(self.binary_fn))

    def calc_parmass(self):
        # compute the particle mass
        parmass = np.diff(self.binary["mass_cum"])
        parmass = np.hstack([self.binary["mass_cum"][0], parmass])

        return parmass

    def calc_quadmom(self):
        # compute the 2nd-order time derivative of mass quadrupole moment
        parmass = self.calc_parmass()  # particle mass

        x  = self.binary["x"]
        y  = self.binary["y"]
        z  = self.binary["z"]

        vx = self.binary["vx"]
        vy = self.binary["vy"]
        vz = self.binary["vz"]

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
        self.par = dict()

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
                    print("Unknown parameter setting in {}: {}".format(fn, expr))
                    continue

                # store to self.par
                self.par[varname.lower()] = eval(varname)  # use keyword in lower case

    def getpar(self, key):
        """
        Return value stored in 'par'. The 'key' can be either upper or lower case.

        If no data found, return None
        """
        return self.par.get(key.lower(), None)


class sphynx(sphynx_loader, sphynx_binary, sphynx_par):
    """
    Contains functions for visualization
    """
    def __init__(self, rundir, runpath):
        sphynx_loader.__init__(self, rundir = rundir, runpath = runpath)
        sphynx_binary.__init__(self, rundir = rundir, runpath = runpath)
        sphynx_par.__init__(self, rundir = rundir, runpath = runpath)
        self.path = os.path.join(runpath, rundir)

        self.get_bounce()
        self.get_tiempos()

    def __str__(self):
        print(self.path)

    def __get_evolution(self, quant, tscale = "pb"):
        """
        Get the evolution of given quantity 'quant'. Currently, only support quantities in

            central_values.d
            luminosity.d
            radgrav.d
        """
        if quant in quant_central:
            if self.central_values is None:
                self.get_central_values()

            time = self.central_values["t"]
            data = self.central_values[quant]

        elif quant in quant_lumin:
            if self.luminosity is None:
                self.get_luminosity()

            time = self.luminosity["t"]
            data = self.luminosity[quant]

        elif quant in quant_radgrav:
            if self.radgrav is None:
                self.get_radgrav()

            time = self.radgrav["t"]
            data = self.radgrav[quant]

        else:
            raise ValueError("Unknown quantity: {}".format(quant))

        if tscale == "pb":
            time = time - self.pb_time

        return time, data

    def getprofile(self, nout):
        # get the data in the binary file in directory 'data'
        fn = os.path.join("data", "s1.{:06d}".format(nout))
        self.get_binary(fn)

    def getinitmodel(self):
        # get the data in the initial model 'archivo'
        fn = self.getpar('archivo')
        fn = os.path.join("..", "initialmodels", "fn")  # path relative to the run directory
        self.get_binary(fn)

    def get_ytobj(self, nout, n_ref = 64):
        # create the yt object from the binary file in directory 'data'
        fn = os.path.join("data", "s1.{:06d}".format(nout))
        self.create_ytobj(fn, n_ref)

    def _get_xlabel(self, tscale = "pb"):
        # choose the x label
        if self.pb and tscale == "pb":
            return r"$t_\mathrm{pb}$ [ms]"
        else:
            return r"Time [ms]"

    def _get_gw_text(self, mode):
        assert mode in ["plus", "cross", "both"], "Unknown mode: {}".format(mode)

        if mode == "plus":
            return r"$h_\plus$"
        elif mode == "cross":
            return r"$h_\mathrm{x}$"
        else:
            return r"$h$"

    def _get_gw_ylabel(self, mode):
        # choose the y label for gw plot
        return r"$10^{21}$" + self._get_gw_text(mode)

    def _calc_gw_strain(self, phi, theta, dist = 10, mode = "both"):
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
        if not isinstance(nouts, (tuple, list)):
            nouts = nouts,

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
                self.getprofile(nout)
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

        print("Output file: {}".format(fnout))

    def plot_evolution(self, quant, tscale = "pb", ax = None, labels = None, **kwargs):
        """
        Plot the evolution of quantity.

        Parameters
        ----------
        tscale: "pb": show the time scale relative to t_pb
                Otherwise: show the physical time
        """
        if not isinstance(quant, (tuple, list)):
            quant = quant,

        if labels is None:
            labels = quant
        elif not isinstance(labels, (tuple, list)):
            labels = labels,

        # load data
        data_all = [self.__get_evolution(q, tscale)  for q in quant]

        # plot
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

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
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

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
        gw_strain = self._calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)

        # plot
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

        ax.plot((self.radgrav["t"] - self.pb_time) * sec2ms, gw_strain * 1e21, **kwargs)

        label_x = self._get_xlabel()
        label_y = self._get_gw_ylabel(mode)
        ax.set_xlabel(label_x)
        ax.set_ylabel(label_y)
        ax.set_title(r"$\phi = {:.2f} \pi, \theta = {:.2f} \pi$".format(phi / np.pi, theta / np.pi))

        if fig is not None:
            return fig, ax

    def plot_gw_spectrum(self, phi, theta, dist = 10, mode = "both", method = "moore14",
                         dt = None, tstart = None, tend = None, ax = None, **kwargs):
        """
        Plot the spectra of GW strains
        """
        assert method in ["kcpan", "moore14", "periodogram"], "Unknown method: {}.".format(method)

        gw_strain = self._calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)
        gw_time   = self.radgrav["t"] - self.pb_time

        if method == "periodogram":
            freq, hchar  = calc_spectrum_periodogram(gw_time, gw_strain,
                                                    tstart = tstart, tend = tend, dt = dt)
        else:
            freq, hchar  = calc_spectrum_fft(gw_time, gw_strain,
                                            method = method, dist = dist,
                                            tstart = tstart, tend = tend, dt = dt)

        # plot
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

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

        gw_strain = self._calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)
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
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = None

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

        gw_strain = self._calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)
        gw_time   = self.radgrav["t"] - self.pb_time

        gw_strain = self._calc_gw_strain(phi = phi, theta = theta, dist = dist, mode = mode)

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
#        ax2.set_yticks([0, 1000, 2000, 3000])

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

