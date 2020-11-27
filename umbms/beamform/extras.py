"""
Tyson Reimer
University of Manitoba
November 7, 2018
"""

import numpy as np

###############################################################################


def get_scan_freqs(ini_f, fin_f, n_freqs):
    """Returns the linearly-separated frequencies used in a scan

    Returns the vector of frequencies used in the scan.

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    scan_freq_vector : array_like
        Vector containing each frequency used in the scan
    """

    # Get the vector of the frequencies used
    scan_freq_vector = np.linspace(ini_f, fin_f, n_freqs)

    return scan_freq_vector


def get_freq_step(ini_f, fin_f, n_freqs):
    """Gets the incremental frequency step df in Hz used in the scan

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    df : float
        The incremental frequency step-size used in the scan, in Hz
    """

    # Get the vector of the frequencies used in the scan
    freqs = get_scan_freqs(ini_f, fin_f, n_freqs)

    # Find the incremental frequency step-size
    df = freqs[1] - freqs[0]

    return df


def get_scan_times(ini_f, fin_f, n_freqs):
    """Returns the times-of-response obtained after using the IDFT

    Gets the vector of time-points for the time-domain representation
    of the radar signals, when the IDFT is used to convert the data
    from the frequency to the time-domain.

    Parameters
    ----------
    ini_f : float
        The initial frequency, in Hz, used in the scan
    fin_f : float
        The final frequency, in Hz, used in the scan
    n_freqs : int
        The number of frequency points used in the scan

    Returns
    -------
    scan_times : array_like
        Vector of the time-points used to represent the radar signal
        when the IDFT is used to convert the data from the
        frequency-to-time domain, in seconds
    """

    # Get the incremental frequency step-size
    freq_step = get_freq_step(ini_f, fin_f, n_freqs)

    # Compute the corresponding time-domain step-size when the IDFT is
    # used to convert the measured data to the time domain
    time_step = 1 / (n_freqs * freq_step)

    # Get the vector of the time-points used to represent the
    # IDFT-obtained signal
    scan_times = np.linspace(0, n_freqs * time_step, n_freqs)

    return scan_times
