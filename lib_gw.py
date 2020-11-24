#  Purpose:
#   Compute the GW emission amplitude of the two polarization modes, A_+ and A_x,
#   using the 2nd time derivative of mass quadrupole tensor, in cartesian coordinate,
#   from the GAMER code.
#
#  Reference:
#    Scheidegger et al. 2008, A&A, 490, 231
#    Andresen et al. 2017, MNRAS, 468, 2032
#
# Last Update: 2020/11/03
# He-Feng Hsieh


import numpy as np
from scipy.fftpack import fft
from scipy.interpolate import interp1d
from scipy.signal import stft, periodogram


# physcial constant
const_G = 6.67428e-8
const_c = 2.99792e10
kpc2cm  = 3.086e21    # kpc in units of cm
sec2ms  = 1e3         # second in units of ms


def conv_cart2sph(Ixx, Ixy, Ixz, Iyy, Iyz, Izz, phi, theta):
    """
    Use symbol I for \ddot{Q}.

    Compute the component I_pp, I_tp, and I_tt from the second-order time derivative
    of mass quadrupole moment in Cartesian coordinate.

    Parameters
    ----------
    phi: azimuthal angle
    theta: polar angle
    """

    sin_phi   = np.sin(phi)
    cos_phi   = np.cos(phi)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    Ipp = Ixx * sin_phi**2 + Iyy * cos_phi**2 - 2 * Ixy * sin_phi * cos_phi

    Itp = (Iyy - Ixx) * cos_theta * sin_phi * cos_phi \
        + Ixy * cos_theta * (cos_phi**2 - sin_phi**2) \
        + Ixz * sin_theta * sin_phi \
        - Iyz * sin_theta * cos_phi

    Itt = (Ixx * cos_phi**2 + Iyy * sin_phi**2 + 2 * Ixy * sin_phi * cos_phi) * cos_theta**2 \
        + Izz * sin_theta**2 \
        - 2 * (Ixz * cos_phi + Iyz * sin_phi) * sin_theta * cos_theta

    return Ipp, Itp, Itt


def calc_amplitude(Ipp, Itp, Itt):
    """
    Compute the amplitude of polarization mode
    """
    A_plus = Itt - Ipp
    A_cross = 2 * Itp

    return A_plus, A_cross


def calc_strain(A_plus = 0.0, A_cross = 0.0, dist = 10):
    """
    Compute the GW strains:

         TT        1
        h    = ---------- ( A  * e  + A  * e  )
         ij     distance     +    +    x    x

    Parameters
    ----------
    A_plus:  amplitude of A_plus mode
    A_cross: amplitude of A_cross mode
    dist:    distance in kpc
    """
    # convert distnace in units of cm
    dist *= kpc2cm

    return (A_plus + A_cross) / dist


def set_timerange(time, tstart = None, tend = None):
    """
    assign the tstart = time[0] and tend = time[-1] if not specified
    otherwise, compare the tstart and tend with time, and reset their values if out of domain

    used in following functions that calculate the spectrum and spectrogram of gw
    """
    # assume the input time increases monotonically
    t_min = time[0]
    t_max = time[-1]

    if tstart is None:
        tstart = t_min
    else:
        if tstart < t_min:
            tstart = t_min
            print("Input tstart is earlier than the available data. Reset it to {}.".format(tstart))

    if tend is None:
        tend = t_max
    else:
        if tend > t_max:
            tend = t_max
            print("Input tend is beyond the available data. Reset it to {}.".format(tend))

    return tstart, tend


def calc_spectrum_fft(data_time, data_strain, dist = 10,
                      tstart = None, tend = None, dt = None,
                      method = "moore14"):
    """
    Compute the spectrum of GW characteristic strain using FFT

    Parametrs
    ---------
    data_time: physical time in sec
    data_strain: GW strain
    dist:  distance in kpc
    tstart: the start time of the spectrogram in sec
    tend: the final time of the spectrogram in sec
    dt: time interval between interpolated data, in sec
    method: method for calculating the characteristic strain
            'kcpan'  : method in KC's scripts
            'moore14': eq. (17) in Moore et al., 2015, Gravitational-wave sensitivity curves
    """
    assert method in ["kcpan", "moore14"], "Unknown method: {}.".format(method)

    ### set up parameters
    tstart, tend = set_timerange(time = data_time, tstart = tstart, tend = tend)
    data_time_min, data_time_max = np.min(data_time), np.max(data_time)

    # remove unit in GW strain
    if method == "kcpan":
        data_strain = data_strain / (const_G / const_c**4 / (dist * kpc2cm))

    # compute the spectra
    window = tend - tstart

    if dt is None:
        dt = window / 1e5

    numpt      = int(window / dt)
    xf         = np.linspace(0, 1 / (2 * dt), numpt // 2)
    selected_t = np.linspace(tstart, tstart + numpt * dt, numpt)
    selected_h = np.zeros(numpt)

    # function for uniform sampling using linear interpolation
    func_inte = interp1d(data_time, data_strain)

    cond_indomain = (data_time_min < selected_t) & (selected_t < data_time_max)
    selected_h[cond_indomain] = func_inte(selected_t[cond_indomain])


    ### do FFT, and store the frequency up to 1 / (2 * dt)
    yf = fft(selected_h)
    yf = np.abs(yf[:numpt // 2]) * (2. / numpt)

    # convert units
    if method == "kcpan":
        dEdf  = 0.6 * const_G / const_c**5 * (2 * np.pi * xf)**2 * np.abs(yf)**2
        hchar = np.sqrt(2 / np.pi**2 * const_G / const_c**3 * dEdf ) / (dist * kpc2cm)
    else: # moore14
        hchar = 2 * xf * yf

    ### reduce the data size
    numpt = 3000
    freq_trim  = np.logspace(0, 4.7, numpt)
    hchar_trim = np.zeros(freq_trim.size)
    hchar_trim[:] = np.NaN  # initialize the hchar_trim as a NaN array

    func_inte = interp1d(xf, hchar)

    cond_indomain = (xf[0] < freq_trim) & (freq_trim < xf[-1])
    hchar_trim[cond_indomain] = func_inte(freq_trim[cond_indomain])

    return freq_trim, hchar_trim


def calc_spectrum_periodogram(data_time, data_strain,
                              tstart = None, tend = None, dt = None):
    """
    Compute the spectrum of GW characteristic strain using scipy.signal.periodogram

    The characteristic strain is calculated using Eq. (17) in Moore et al. (2014)

    Parametrs
    ---------
    data_time: physical time in sec
    data_strain: GW strain
    tstart: the start time of the spectrogram in sec
    tend: the final time of the spectrogram in sec
    dt: time interval between interpolated data in sec
    """
    ### set up parameters
    tstart, tend = set_timerange(time = data_time, tstart = tstart, tend = tend)

    if dt is None:
        dt = 1e-5

    numpt = int((tend - tstart) / dt)
    fs    = 1. / dt

    ### interpolation
    func_inte = interp1d(data_time, data_strain)

    times  = np.linspace(tstart, tstart + numpt * dt, numpt)
    values = func_inte(times)

    freq, Pxx = periodogram(values, fs = fs, window = "hann", scaling = "spectrum", detrend = False)

    return freq, 2 * freq * np.sqrt(Pxx)


def calc_spectrogram_fft(data_time, data_strain, window = None, tstart = None, tend = None, numpt = 1000):
    """
    Compute the spectrogram of GW strain

    Parametrs
    ---------
    data_time: physical time in sec
    data_strain: GW strain
    window: width of the window function in ms
    tstart: the start time of the spectrogram in sec
    tend: the final time of the spectrogram in sec
    numpt: number of points in the time domain [t - window, t + window]
    """
    ### set up parameters
    tstart, tend = set_timerange(time = data_time, tstart = tstart, tend = tend)

    if window is None:
        window = 10.

    data_time_min, data_time_max = np.min(data_time), np.max(data_time)

    window = window * 1.e-3  # in sec
    dt     = window / numpt  # in sec
    freq   = np.linspace(0, 1 / (2 * dt), numpt // 2)

    # compute the spectrogram at time interval = 0.5 ms
    times = np.linspace(tstart, tend, int((tend - tstart) * 1e3 * 2))  # in sec
    spectrogram = np.zeros([times.size, freq.size])

    # function for linear interpolation
    func_inte = interp1d(data_time, data_strain)


    ### compute here
    for idx, time in enumerate(times):
        t1 = time - 0.5 * window
        t2 = time + 0.5 * window

        selected_t = np.linspace(t1, t2, numpt)
        selected_h = np.zeros(numpt)

        # interpolate the input data_strain to fill up selected_h
        cond_indomain = (data_time_min < selected_t) & (selected_t < data_time_max)
        selected_h[cond_indomain] = func_inte(selected_t[cond_indomain])

        # do FFT, and store the frequency up to 1 / (2 * dt)
        if t2 < data_time_min or t1 > data_time_max:
            spectrogram[idx] = np.nan
        else:
            freq_fft = fft(selected_h)
            spectrogram[idx] = np.abs(freq_fft[:numpt // 2]) * (2. / numpt)

    return times, freq, spectrogram


def calc_spectrogram_stft(data_time, data_strain, window = None, tstart = None, tend = None, dt = 1e-5, **kwargs):
    """
    Compute the spectrogram of GW strain using stft

    Parametrs
    ---------
    data_time: physical time in sec
    data_strain: GW strain
    window: Length of each segment.
    tstart: the start time of the spectrogram in sec
    tend: the final time of the spectrogram in sec
    dt: time separation between measurement values in sec
    """
    ### set up parameters
    tstart, tend = set_timerange(time = data_time, tstart = tstart, tend = tend)

    if window is None:
        window = 256

    numpt = int((tend - tstart) / dt)
    fs    = 1. / dt

    # interpolate the measurement value linearly for fft
    func_inte = interp1d(data_time, data_strain)

    times = np.linspace(tstart, tstart + numpt * dt, numpt)
    value = func_inte(times)

    # compute here
    f, t, Zxx = stft(value, fs = fs, nperseg = window)

    return t + tstart, f, np.abs(Zxx)

