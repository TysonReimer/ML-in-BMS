"""
Tyson Reimer
University of Manitoba
July 10th, 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.cm import get_cmap

from sklearn.decomposition import PCA

from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import load_pickle, save_pickle
from umbms.ai.preproc import to_td

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/')

__OUT_DIR = os.path.join(get_proj_path(), 'output/figs/')
verify_path(__OUT_DIR)

fig_out = os.path.join(get_proj_path(), 'output/figs/')
verify_path(fig_out)

###############################################################################


def get_valid_session_ns(metadata, dataset):
    """Gets the session numbers in the metadata for more than 5 scans

    Parameters
    ----------
    metadata : array_like
        Array of metadata info
    dataset : array_like
        Measured data

    Returns
    -------
    valid_session_ns : list
        List of the session IDs of all sessions with more than 5
        scans

    """

    # Get the Session ID of each scan
    session_ids = np.array([md['n_session'] for md in metadata])

    # Find the unique Session IDs
    unique_session_ids = np.unique(session_ids)

    valid_session_ns = []  # Init list to return

    for ii in unique_session_ids:  # For each unique session ID

        tar_idxs = session_ids == ii  # Target indices in metadata

        # Number of scans in this session
        n_here = len(dataset[tar_idxs, :, :])

        if n_here >= 5:  # If the number of scans is >= 5
            valid_session_ns.append(ii)  # Store it

    return valid_session_ns


def get_total_n_sinos(ses_ids, dataset, all_ses_ids):
    """Find the total number of sinograms from all sessions

    Parameters
    ----------
    ses_ids : list
        List of session IDs
    dataset : array_like
        Sinogram dataset
    all_ses_ids : array_like
        Session ID of all samples in dataset

    Returns
    -------
    total_n : int
        Total number of sinograms from the specified sessions
    """

    total_n = 0  # Init counter

    for n_ses in ses_ids:  # For each session ID

        tar_idxs = all_ses_ids == n_ses  # Indices of target session

        ses_sinos = dataset[tar_idxs, :, :]  # Sinos from this session
        total_n += np.size(ses_sinos, axis=0)  # Add to counter

    return total_n


def get_sino_summaries(dataset, metadata):
    """Gets the sinogram summaries

    Parameters
    ----------
    dataset : array_like
        Sinogram dataset
    metadata : array_like
        Metadata array for each sample

    Returns
    -------
    sino_summaries : array_like
        The average and standard deviation of the signals for each
        sinogram
    sino_md : array_like
        Metadata for the scans from which the average and stdev
        signals were obtained
    """

    # Get the session IDs to include only sessions with 5 or more scans
    ses_ids = get_valid_session_ns(metadata=metadata, dataset=dataset)

    # Get the session ID of all scans
    all_ses_ids = np.array([md['n_session'] for md in metadata])

    # Find the number of sinograms that correspond to the specified
    # sessions
    n_sinos = get_total_n_sinos(ses_ids=ses_ids, all_ses_ids=all_ses_ids,
                                dataset=dataset)

    # Init arrays to return
    sino_summaries = np.zeros([n_sinos, 35, 2])
    sino_md = []

    cc = 0  # Init counter

    for n_ses in ses_ids:  # For each session ID

        tar_idxs = all_ses_ids == n_ses  # Target indices

        # Get the sinograms and metadata from this session
        ses_sinos = dataset[tar_idxs, :, :]
        ses_md = metadata[tar_idxs]

        # Get the number of sinograms in this session
        n_sinos = np.size(ses_sinos, axis=0)

        for sino_idx in range(n_sinos):  # For each sinogram

            # Get the average and standard deviation radar signal
            avg_sig = np.mean(ses_sinos[sino_idx, :, :], axis=1)
            std_err = (np.std(ses_sinos[sino_idx, :, :], axis=1))

            # Store results
            sino_summaries[cc, :, 0] = avg_sig
            sino_summaries[cc, :, 1] = std_err
            sino_md.append(ses_md[sino_idx])

            cc += 1

    return sino_summaries, sino_md


def save_sino_summaries(gen='g2'):
    """Create and save the sinogram summaries

    Parameters
    ----------
    gen : str
        Must be in ['g1', 'g2'], string to identify which data gen
    """

    # Load the data and metadata
    if gen in ['g1']:

        data = load_pickle(os.path.join(__DATA_DIR, 'g1/g1_fd.pickle'))
        metadata = load_pickle(os.path.join(__DATA_DIR,
                                            'g1/g1_metadata.pickle'))

    else:  # If Gen-2

        data = load_pickle(os.path.join(__DATA_DIR, 'g2/g2_fd.pickle'))
        metadata = load_pickle(os.path.join(__DATA_DIR,
                                            'g2/g2_metadata.pickle'))

    # Convert to the time-domain
    data = np.abs(to_td(data))
    metadata = np.array(metadata)

    # Get the sinogram summaries (average and stdev signals)
    sino_summaries, sino_md = get_sino_summaries(dataset=data,
                                                 metadata=metadata)

    # Save to .pickle files
    save_pickle(sino_summaries,
                os.path.join(__OUT_DIR, '%s_sino_summaries.pickle' % gen))
    save_pickle(sino_md,
                os.path.join(__OUT_DIR, '%s_sino_summaries_md.pickle' % gen))


def load_sino_summaries(gen='g2'):
    """Loads sinogram summaries from .pickle files

    Parameters
    ----------
    gen : str
        Must be in ['g1', 'g2'], string to identify which data gen

    Returns
    -------
    sino_summaries : array_like
        The average and standard deviation of the signals for each
        sinogram
    sino_md : array_like
        Metadata for the scans from which the average and stdev
        signals were obtained
    """

    summaries = load_pickle(os.path.join(__OUT_DIR,
                                         '%s_sino_summaries.pickle' % gen))
    metadata = load_pickle(os.path.join(__OUT_DIR,
                                        '%s_sino_summaries_md.pickle' % gen))

    return summaries, metadata


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Create and save the sinogram summaries
    save_sino_summaries(gen='g2')

    # Load them to memory
    g2_sigs, g2_md = load_sino_summaries(gen='g2')

    pca = PCA()  # Init PCA model

    # Perform PCA on mean signals
    g2_sigs = pca.fit_transform(g2_sigs[:, :, 0])

    # Init plot
    viridis = get_cmap('inferno')
    plt.figure(figsize=(16,10))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.gca().set_aspect('equal', adjustable='box')

    # **** PLOT BY ADI ID
    adi_ids = np.array([md['phant_id'].split('F')[0] for md in g2_md])

    # Hard-code list in-order from smallest adipose shell to largest
    unique_adi_ids = ['A1', 'A11', 'A12', 'A13', 'A14', 'A2', 'A15',
                      'A16', 'A3']
    n_ses = np.size(unique_adi_ids, axis=0)
    cc = 0

    for adi_id in unique_adi_ids:  # For each adipose ID

        # Get the avg and stdev signals here
        sigs_here = g2_sigs[adi_ids == adi_id, :]

        # Plot all the average signals
        plt.scatter(x=sigs_here[:, 0], y=sigs_here[:, 1],
                    color=viridis(cc / (n_ses )), marker='o', s=4,
                    label='%s' % adi_id)

        # Get the avg and stdev signal of all scans from this adipose
        # shell
        mean_sig = np.mean(sigs_here, axis=0)
        std_sig = np.std(sigs_here, axis=0)

        # Plot an ellipse to illustrate the centroid of the scans
        # of phantoms with this adipose shell
        ellipse = Ellipse(xy=(mean_sig[0], mean_sig[1]),
                          width=2 * std_sig[0], height=2 * std_sig[1],
                          alpha=0.2,
                          color=viridis(cc / n_ses))
        plt.gcf().gca().add_artist(ellipse)

        plt.text(x=mean_sig[0], y=mean_sig[1],
                 s='%s' % adi_id,
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'white', 'alpha': 0.8})
        cc += 1

    plt.legend(ncol=2, fontsize=14)

    plt.xlabel('First Principal Component', fontsize=22)
    plt.ylabel('Second Principal Component', fontsize=22)

    plt.show()
    plt.savefig(os.path.join(fig_out, 'pca_by_adi.png'),
                transparent=False, dpi=600, bbox_inches='tight')
