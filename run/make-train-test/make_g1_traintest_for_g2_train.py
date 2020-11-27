"""
Tyson Reimer
University of Manitoba
February 22nd, 2020
"""

import os
import numpy as np

from umbms import get_proj_path, get_script_logger, verify_path

from umbms.loadsave import load_pickle, save_pickle
from umbms.ai.makesets import shuffle_arrs, get_class_labels

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/g1/')

__OUT_DIR = os.path.join(get_proj_path(), 'data/umbmid/g1-train-test/')
verify_path(__OUT_DIR)

###############################################################################


def report_md_breakdown(metadata):
    """Prints a breakdown of metadata info to logger

    Parameters
    ----------
    metadata : array_like
        List of dicts of the metadata
    """

    n_samples = len(metadata)  # Find number of samples in metadata

    # Find the adipose shell ID of each sample
    adi_ids = np.array([md['phant_id'].split('F')[0] for md in metadata])
    unique_adi_ids = np.unique(adi_ids)  # Find the unique adi IDs

    logger.info('\tAdipose Fractions:')
    for adi_id in unique_adi_ids:

        # Find fraction samples with this adipose ID
        frac_here = np.sum(adi_ids == adi_id) / n_samples
        logger.info('\t\t%s:\t%.2f%%' % (adi_id, 100 * frac_here))

    # Get tumor presence and print  to logger
    tum_presence = get_class_labels(metadata)
    logger.info('\tTumor Fraction:\t%.2f'
                % (np.sum(tum_presence) * 100 / n_samples))

    # Get BI-RADS class of each sample
    birads_classes = np.array([md['birads'] for md in metadata])
    unique_birads = np.unique(birads_classes)

    logger.info('\tBI-RADS Fractions:')
    for birads_class in unique_birads:

        # Find fraction of samples with this BI-RADS class
        frac_here = np.sum(birads_classes == birads_class) / n_samples
        logger.info('\t\tClass %d:\t%.2f' % (birads_class, 100 * frac_here))


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load all Gen-1 frequency-domain data and metadata
    fd_data = load_pickle(os.path.join(__DATA_DIR, 'g1_fd.pickle'))
    metadata = load_pickle(os.path.join(__DATA_DIR, 'g1_metadata.pickle'))

    # Init vars for train/test fractions
    tr_frac = 0
    te_frac = 0

    # Until the train tumor fraction is approx 50%
    while not (0.45 <= tr_frac <= 0.55):

        # Shuffle the arrays
        [fd_data, metadata] = shuffle_arrs([fd_data, metadata])

        # Split data/metadata into train and test sets
        tr_data = fd_data[:125, :, :]
        tr_md = metadata[:125]
        te_data = fd_data[125:, :, :]
        te_md = metadata[125:]

        # Get the train/test class labels to determine tumor fraction
        cv_labels = get_class_labels(tr_md)
        test_labels = get_class_labels(te_md)

        # Get tumor fraction in the training set
        tr_frac = np.sum(cv_labels) / len(cv_labels)

    # Save train/test sets as .pickles
    save_pickle(te_data, os.path.join(__OUT_DIR, 'test_fd.pickle'))
    save_pickle(te_md, os.path.join(__OUT_DIR, 'test_md.pickle'))
    save_pickle(tr_data, os.path.join(__OUT_DIR, 'train_fd.pickle'))
    save_pickle(tr_md, os.path.join(__OUT_DIR, 'train_md.pickle'))
