#! /usr/bin/env python
#
# -*- coding: utf-8 -*-
#
#  Purpose:
#    Format of data generated in SPHYNX simulations
#
#  Last Updated: 2020/10/14
#  He-Feng Hsieh


import numpy as np


# format in tiempos.d
# Caution: time_rel (time relative core bounce) is not correct after restart
fmt_tiempos = np.dtype([("idx",      "<i4"),
                        ("time",     "<f8"),
                        ("time_rel", "<f8")  ])


# format in central_values.d
fmt_central_values = np.dtype([("t",    "<f8"),
                               ("dens", "<f8"),
                               ("ye",   "<f8"),
                               ("entr", "<f8"),
                               ("yl",   "<f8"),
                               ("temp", "<f8")  ])


# format in radgrav.d
fmt_radgrav = np.dtype([("idx",     "<i4"),
                        ("t",       "<f8"),
                        ("tiempo",  "<f8"),
                        ("h_plus",  "<f8"),
                        ("h_cross", "<f8"),
                        ("Ixx",     "<f8"),
                        ("Iyy",     "<f8"),
                        ("Ixy",     "<f8"),
                        ("Ixz",     "<f8"),
                        ("Iyz",     "<f8"),
                        ("Izz",     "<f8")  ])

# format in luminosity.d
fmt_luminosity = np.dtype([("t",        "<f8"),
                           ("Lnu_e",    "<f8"),
                           ("Lnu_ebar", "<f8"),
                           ("Lnu_x",    "<f8"),
                           ("Nnu_e",    "<f8"),
                           ("Nnu_ebar", "<f8")  ])

# format in binary files: initial condition & data/
fmt_binary = np.dtype([("dummy1",     "<i4"),
                       ("idx",        "<i4"),  # particle ID
                       ("x",          "<f8"),  # position in the x direction
                       ("y",          "<f8"),  # position in the y direction
                       ("z",          "<f8"),  # position in the z direction
                       ("h",          "<f8"),  # smoothing length
                       ("u",          "<f8"),  # specific internal energy
                       ("promro",     "<f8"),  # density
                       ("vx",         "<f8"),  # velocity in the x direction
                       ("vy",         "<f8"),  # velocity in the y direction
                       ("vz",         "<f8"),  # velocity in the z direction
                       ("radius",     "<f8"),  # radius
                       ("ye",         "<f8"),  # Ye
                       ("mass_cum",   "<f8"),  # cumulative mass
                       ("p",          "<f8"),  # pressure
                       ("ugrav",      "<f8"),
                       ("omega",      "<f8"),
                       ("fx",         "<f8"),  # gravitational force in the x direction (with GREP correction if any)
                       ("fy",         "<f8"),  # gravitational force in the y direction (with GREP correction if any)
                       ("fz",         "<f8"),  # gravitational force in the z direction (with GREP correction if any)
                       ("gradp_x",    "<f8"),  # pressure gradient in the x direction
                       ("gradp_y",    "<f8"),  # pressure gradient in the y direction
                       ("gradp_z",    "<f8"),  # pressure gradient in the z direction
                       ("temp",       "<f8"),
                       ("avisc",      "<f8"),  # artificial viscosity?
                       ("energy",     "<f8"),
                       ("nuedot",     "<f8"),
                       ("s",          "<f8"),  # entropy
                       ("vrad",       "<f8"),  # radial velocity
                       ("ynu_e",      "<f8"),
                       ("ynu_ebar",   "<f8"),
                       ("znu_e",      "<f8"),
                       ("znu_ebar",   "<f8"),
                       ("nvi",        "<f8"),
                       ("gprad",      "<f8"),  # pressure gradient in the r direction
                       ("fgrad",      "<f8"),  # gravitational force in the r direction (with GREP correction if any)
                       ("c",          "<f8"),  # sound speed
                       ("Lnur_e",     "<f8"),
                       ("Lnur_ebar",  "<f8"),
                       ("Lnuxr",      "<f8"),
                       ("Lnuac_e",    "<f8"),
                       ("Lnuac_ebar", "<f8"),
                       ("Lnuxac",     "<f8"),
                       ("yedot",      "<f8"),
                       ("dummy2",     "<i4")  ])
