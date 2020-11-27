"""
Tyson Reimer
University of Manitoba
February 2nd, 2020
"""

import numpy as np

###############################################################################


def get_class_labels(metadata):
    """Gets the binary class labels from a set of metadata

    Parameter
    ---------
    metadata : list
        List containing the metadata dicts for each sample

    Returns
    -------
    labels : array_like
        Array of the binary class labels for each sample
    """

    labels = np.array([~np.isnan(md['tum_rad']) for md in metadata])

    return labels


def shuffle_arrs(arr_list, return_seed=False,
                 rand_seed=0):
    """Shuffle arrays to maintain inter-array ordering

    Shuffles each array in the list of arrays, arrays_list, such that
    the inter-array order is maintained (i.e., the zeroth element of
    the all arrays before shuffling corresponds to the nth element of
    all arrays after shuffling)

    Parameters
    ---------
    arr_list : list
        List containing each array that will be shuffled
    rand_seed : int
        The seed to use for shuffling each array
    return_seed : bool
        If True, will return the seed used to shuffle the arrays (for
        reproducibility)

    Returns
    -------
    shuffled_arrs : list
        List containing the shuffled arrays
    rand_seed :
        The seed that was used to shuffle the arrays
    """

    shuffled_arrs = []  # Init arr for storing the shuffled arrays

    for array in arr_list:  # For each array in the list
        np.random.seed(rand_seed)  # Set the seed

        if type(array) == list:  # If the 'array' is actually a list

            # Copy the list into a new var that will be shuffled
            shuffled_arr = [ii for ii in array]

        else:  # If the array is an array

            # Make a copy that will be shuffled
            shuffled_arr = array * np.ones_like(array)

        np.random.shuffle(shuffled_arr)  # Shuffle the array

        # Append the shuffled array to the list of shuffled arrays
        shuffled_arrs.append(shuffled_arr)

    if return_seed:  # If returning the seed, then do so
        return shuffled_arrs, rand_seed
    else:
        return shuffled_arrs
