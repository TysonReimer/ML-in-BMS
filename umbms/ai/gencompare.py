"""
Tyson Reimer
University of Manitoba
June 25th, 2020
"""

import numpy as np

###############################################################################


def correct_g1_ini_ant_ang(data):
    """Corrects Gen-1 data that started at -102.5deg to start at -130deg

    Parameters
    ----------
    data : array_like
        Gen-1 data, ini_ant_ang=-102.5deg

    Returns
    -------
    cor_data : array_like
        Gen-1 data, ini_ant_ang=-130.0deg
    """

    wide_data = np.concatenate((data, data, data), axis=2)

    cor_data = np.zeros_like(data)

    for ii in range(np.size(data, axis=2)):
        cor_data[:, :, ii] = wide_data[:, :, ii + 5]

    return cor_data
