"""
Tyson Reimer
University of Manitoba
February 13th, 2020
"""

import numpy as np

###############################################################################


def aug_hor_ref(data, metadata):
    """Horizontally-reflects samples to generate augmented samples

    Performs data augmentation to obtain horizontally reflected samples.

    Parameters
    ----------
    data : array_like
        The dataset features for every sample
    metadata : array_like
        The metadata for every sample in the dataset

    Returns
    -------
    reflected_data : array_lke
        The dataset containing the features for every original sample,
        and the augmented (reflected) samples
    metadata : array_like
        The IDs of every original sample and the augmented samples
    """

    # Initialize arrays for storing the augmented data, labels, and IDs
    reflected_data = np.zeros([np.size(data, axis=0) * 2,
                               np.size(data, axis=1),
                               np.size(data, axis=2)],
                              dtype=data.dtype)

    reflected_metadata = []

    # For every sample in the dataset
    for sample in range(np.size(data, axis=0)):
        this_sample_data = data[sample, :, :]  # Get this sample

        # Keep the original sample and a horizontal flip of the sample
        reflected_data[sample, :, :] = this_sample_data
        reflected_data[-(sample + 1), :, :] = np.flip(this_sample_data,
                                                      axis=1)

        # Store the metadata for this sample
        reflected_metadata.append(metadata[sample])

    # Get the IDs for all the samples
    reflected_metadata = np.array(reflected_metadata)
    reflected_metadata = np.concatenate((reflected_metadata,
                                         np.flip(reflected_metadata, axis=0)))

    return reflected_data, reflected_metadata


def aug_hor_translate(data, metadata, step_size=10):
    """Horizontally-translates samples to generate augmented samples

    Performs data augmentation to obtain horizontally translated samples.

    Parameters
    ----------
    data : array_like
        The dataset features for every sample
    metadata : array_like
        The metadata for every sample in the dataset
    step_size : int
        The step-size to be used for horizontal translations

    Returns
    -------
    reflected_data : array_lke
        The dataset containing the features for every original sample,
        and the augmented (translated) samples
    reflected_metadata : array_like
        The IDs of every original sample and the augmented samples
    """

    # Find the number of steps that can be done
    num_steps = (np.size(data, axis=2) // step_size - 1)

    # Initialize arrays for storing the augmented data, labels, and IDs
    translated_data = np.zeros([np.size(data, axis=0) * num_steps,
                                np.size(data, axis=1),
                                np.size(data, axis=2)],
                               dtype=data.dtype)

    translated_metadata = []

    # For every sample in the set
    for sample in range(np.size(data, axis=0)):
        this_sample = data[sample, :, :]  # Get the sample features here
        for step in range(num_steps):  # For every translation step we can do
            translated_sample = np.zeros_like(this_sample)

            # Obtain the translated sample for this translation step by
            # performing a translation
            for ii in range(np.size(this_sample, axis=1)):

                translated_sample[:, ii] = \
                    this_sample[:, ii - step_size * step]

            # Add this translated sample to the augmented dataset
            # (and labels, IDs arrays)
            translated_data[sample * num_steps + step, :, :] = \
                translated_sample

            translated_metadata.append(metadata[sample])

    # Convert the IDs from a list to an arr
    translated_metadata = np.array(translated_metadata)

    return translated_data, translated_metadata


def full_aug(data, metadata, step_size=10):
    """Performs horizontal translation and reflection to generate aug samples

    Performs both horizontal translation and reflection to obtain an
    augmented dataset

    Parameters
    ----------
    data : array_like
        The dataset features for every sample
    metadata : array_like
        The ID tags for every sample in the dataset
    step_size : int
        The step-size to be used for horizontal translations

    Returns
    -------
    data : array_like
        The augmented dataset (features for every sample)
    metadata : array_like
        The augmented labels
    """

    # First, perform augmentation by horizontal translation
    data, metadata = aug_hor_translate(data, metadata, step_size=step_size)

    # Then, perform augmentation by horizontal reflection
    data, metadata = aug_hor_ref(data, metadata)

    return data, metadata
