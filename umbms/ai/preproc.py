"""
Tyson Reimer
University Of Manitoba
June 11th, 2019
"""

import numpy as np

from umbms.beamform.sigproc import iczt

###############################################################################


def to_td(fd_sinos, ini_t=0.5e-9, fin_t=5.5e-9, n_time_pts=35,
          ini_f=1e9, fin_f=8e9):
    """Converts frequency-domain sinograms to the time-domain

    Parameters
    ----------
    fd_sinos : array_like
        3D array, consisting of N frequency-domain sinograms, each
        MxL
    ini_t : float
        The starting time-of-response to be used for computing the ICZT,
        in seconds
    fin_t : float
        The stopping time-of-response to be used for computing the ICZT,
        in seconds
    n_time_pts : int
        The number of points in the time-domain at which the transform
        will be evaluated
    ini_f : float
        The initial frequency used in the scan, in Hz
    fin_f : float
        The final frequency used in the scan, in Hz

    Returns
    -------
    td_sinos : array_like
        Time-domain sinograms
    """

    # Init arr to return
    td_sinos = np.zeros([np.size(fd_sinos, axis=0),
                         n_time_pts, np.size(fd_sinos, axis=2)],
                        dtype=complex)

    # For each frequency-domain sinogram
    for ii in range(np.size(fd_sinos, axis=0)):

        # Convert to the time-domain
        td_sinos[ii, :, :] = iczt(fd_sinos[ii, :, :],
                                  ini_t=ini_t, fin_t=fin_t,
                                  n_time_pts=n_time_pts,
                                  ini_f=ini_f, fin_f=fin_f)

    return td_sinos


def resize_features_for_keras(features):
    """Reshape 3D features for use with Keras

    Reshapes an NxMxL feature set of N samples, each with feature shape MxL,
    for use with keras (reshapes to NxMxLx1)

    Parameters
    ----------
    features : array_like
        An NxMxL arr containing the MxL feature arr for N samples

    Returns
    -------
    features : array_like
        An NxMxLx1 arr containing the MxLx1 feature arr for N samples,
        ready for use in keras models
    """

    features = features.reshape(np.size(features, axis=0),
                                np.size(features, axis=1),
                                np.size(features, axis=2),
                                1)
    return features


def resize_features_for_logreg(features):
    """Reshape 3D features for use with sklearn Logistic Regression

        Parameters
    ----------
    features : array_like
        An NxMxL arr containing the MxL feature arr for N samples

    Returns
    -------
    features : array_like
        An Nx(M*L) arr containing the M*L feature arr for N samples,
        ready for use in keras models
    """

    features = features.reshape(np.size(features, axis=0),
                                np.size(features, axis=1)
                                * np.size(features, axis=2))

    return features
