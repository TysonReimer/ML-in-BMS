"""
Tyson Reimer
University of Manitoba
November 16th, 2020
"""

import os
import numpy as np

import matplotlib.pyplot as plt

import scipy.stats as stats

from umbms import get_proj_path, get_script_logger, verify_path
from umbms.loadsave import load_pickle

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'output/by-adi-preds/')

phant_info_dir = os.path.join(get_proj_path(), 'data/phant-info/')

__OUT_DIR = os.path.join(get_proj_path(), 'output/figs/')
verify_path(__OUT_DIR)

###############################################################################

# Define mid grey color
mid_grey = [118, 113, 113]
mid_grey = [ii / 255 for ii in mid_grey]

# Define light  grey color
light_grey = [217, 217, 217]
light_grey = [ii / 255 for ii in light_grey]

###############################################################################


def get_adi_ids(metadata):
    """Returns a list of the adipose shell IDs for each sample

    Parameters
    ----------
    metadata : array_like
        Array of metadata dicts for each sample

    Returns
    -------
    adi_ids : array_like
        Array of the adipose IDs for each sample in the metadata
    """

    adi_ids = [md['phant_id'].split('F')[0] for md in metadata]

    return adi_ids


def get_adi_id_fracs(metadata):
    """Returns the fraction of samples with each of the adipose IDs

    Parameters
    ----------
    metadata : array_like
        Array of metadata dicts for each sample

    Returns
    -------
    adi_id_fracs : dict
        Dictionary of the fraction of samples in metadata that had
        each of the adipose IDs
    adi_id_errs : dict
        Dictionary of the margin of error of the proportion of samples
        in the metadata that had each of the adipose IDs
    """

    adi_ids = get_adi_ids(metadata)  # Get list of sample adi IDs

    # Specify the unique adipose IDs in order from smallest volume
    # to largest
    unique_ids = ['A1', 'A11', 'A12', 'A13', 'A14', 'A2', 'A15', 'A16', 'A3']

    adi_id_fracs = dict()  # Init dicts to return
    adi_id_errs = dict()

    z_distro = stats.norm()  # Get normal distribution
    z_star = z_distro.ppf(0.95)  # Find 95% C.I. critical value

    for this_id in unique_ids:  # For each adipose ID

        # Find the number of samples with this adipose ID
        n_here = np.sum(np.array([(adi_id in [this_id] and this_id in [adi_id])
                                  for adi_id in adi_ids]))

        # Store the fraction of all samples that had this adipose ID
        adi_id_fracs[this_id] = n_here / len(adi_ids)

        # Store the margin of error here
        adi_id_errs[this_id] = \
            z_star * np.sqrt(adi_id_fracs[this_id]
                             * (1 - adi_id_fracs[this_id]) / len(adi_ids))

    return adi_id_fracs, adi_id_errs


def adi_id_stats(all_preds, wrong_preds):
    """Perform z-test for proportions on test/incorrect prediction sets

    Parameters
    ----------
    all_preds : array_like
        Metadata dict for all samples in the test set
    wrong_preds : array_like
        Metadata dict for all samples in the test set that the
        classifier incorrectly predicted

    Returns
    -------
    pooled_props : dict
        The pooled proportion for each adipose ID
    prop_diffs : dict
        The proportion difference between the all_preds and wrong_preds
        sets, for each adipose ID
    pooled_prop_errs : dict
        The standard error in the pooled proportions, for each
        adipose ID
    ps : dict
        The P-value of the test of the null hypothesis that the
        proportions of the adipose ID in the all_preds and wrong_preds
        are the same, for each adipose ID
    """

    # Get the proportion of each adipose ID in the test set
    all_id_fracs, _ = get_adi_id_fracs(all_preds)
    n_all = len(all_preds)  # Find number of samples in test set

    # Get the proportion of each adipose ID in the set of incorrect
    # predictions
    bad_id_fracs, _ = get_adi_id_fracs(wrong_preds)
    n_bad = len(wrong_preds)

    # Init dicts to return
    ps = dict()
    pooled_props = dict()
    prop_diffs = dict()
    pooled_prop_errs = dict()

    # Get array of unique adipose IDs
    unique_adi_ids = np.unique(np.array(get_adi_ids(all_preds)))

    z_distro = stats.norm()  # Get normal distribution

    for ii in unique_adi_ids:  # For each adipose ID

        # Get the pooled proportion here
        pooled_props[ii] = ((bad_id_fracs[ii] * n_bad
                             + all_id_fracs[ii] * n_all) / (n_bad + n_all))

        # Get the standard error in the pooled proportion
        pooled_prop_errs[ii] = np.sqrt(pooled_props[ii]
                                       * (1 - pooled_props[ii])
                                       * (1 / n_bad + 1 / n_all))

        # Get the z-value of the proportion difference
        z = (bad_id_fracs[ii] - all_id_fracs[ii]) / pooled_prop_errs[ii]

        # Get the value of the proportion difference
        prop_diffs[ii] = bad_id_fracs[ii] - all_id_fracs[ii]

        # Get the two-sided P-value for this comparison
        ps[ii] = 2 * (1 - z_distro.cdf(np.abs(z)))

    return pooled_props, prop_diffs, pooled_prop_errs, ps


def conv_flt_to_sci_not(flt_val):
    """Converts a float or string to sci notation for display

    Parameters
    ----------
    flt_val : float
        The float of the P-value

    Returns
    -------
    str_to_print :
        The string to print
    """

    # If the value is stored in scientific notation
    if 'e' in str(flt_val):

        # Find the exponent
        n_exp = -1 * int(str(flt_val).split('e')[1])

        # Make it a big integer
        flt_val = flt_val * 1e16

        reduce_cc = n_exp  # Store this exponent

    else:  # If the value isn't stored in scientific notation

        reduce_cc = 0  # Store this 'exponent'

    flt_str = str(flt_val)  # Convert to a str

    lead_zero_cc = 0  # Counter for leading zeros
    print_cc = 0  # Counter for number of sig figs to display

    str_to_print = []  # Init list to return

    for ii in range(len(flt_str)):  # For each char in the str

        if print_cc > 3:  # If 3 digits have been stored, break
            break

        # If the char is not a leading zero or a decimal place
        if (flt_str[ii] not in ['0', '.']) or print_cc > 0:

            if print_cc == 0:  # If this is the first sig fig
                str_to_print.append('%s.' % flt_str[ii])
                print_cc += 1  # Increment counter

            # If this is less than the third sig-fig, but not the first
            elif print_cc < 3:
                str_to_print.append('%s' % flt_str[ii])
                print_cc += 1

        else:  # If the char is a leading zero or a decimal place
            if flt_str[ii] in ['0']:
                lead_zero_cc += 1

    # Store the exponent for the P-value
    str_to_print.append((lead_zero_cc + reduce_cc))

    return str_to_print


def plt_adi_id_distros(all_preds, wrong_preds):
    """Plot the distributions of adi IDs in all_preds and wrong_preds

    Parameters
    ----------
    all_preds : array_like
        Array of the metadata of all predictions (the entire test set)
    wrong_preds : array_like
        Array of the metadata of the incorrect predictions (the set
        of all samples for which the classifier made an incorrect
        prediction)
    """

    # Manually specify unique adipose IDs in order from smallest
    # to largest, by volume
    unique_adi_ids = ['A1', 'A11', 'A12', 'A13', 'A14', 'A2', 'A15', 'A16',
                      'A3']

    # Find the fraction of samples that had each adipose ID in the
    # all-prediction set
    all_id_fracs, all_id_errs = get_adi_id_fracs(all_preds)

    # Find the fraction of samples that had each adipose ID in the
    # incorrect-prediction set
    bad_id_fracs, bad_id_errs = get_adi_id_fracs(wrong_preds)

    # Do the z-test for proportions, comparing the proportion in the
    # entire test set to the proportion in the set of incorrect samples
    props, prop_diffs, prop_errs, ps = adi_id_stats(all_preds, wrong_preds)

    xtick_strs = []  # Init list for storing legend strings

    # Init lists for storing data for the plot
    bad_id_frac_errs = []
    all_id_frac_errs = []
    plt_all_id_fracs = []
    plt_bad_id_fracs = []

    for ii in unique_adi_ids:  # For each adipose ID

        # Print the proportion difference and P-value
        print('%4s:\t\t%.2e +/- %.2e\t\tp = %.2e'
              % (ii, prop_diffs[ii], prop_errs[ii], ps[ii]))

        # Store these in a list for plotting
        bad_id_frac_errs.append(bad_id_errs[ii])
        all_id_frac_errs.append(all_id_errs[ii])
        plt_all_id_fracs.append(all_id_fracs[ii])
        plt_bad_id_fracs.append(bad_id_fracs[ii])

        if ps[ii] == 0:  # If the P-value was zero

            # Manually set string
            xtick_strs.append(r'P < 1$\mathdefault{\times10^{-16}}$')

        else:  # If the P-value was nonzero

            # Get the P-value string for display purposes
            to_print = conv_flt_to_sci_not(ps[ii])

            # Append this to the list of xtick labels
            xtick_strs.append(r'P = %s%s%s$\mathdefault{\times10^{-%d}}$'
                              % (to_print[0], to_print[1], to_print[2],
                                 to_print[3]))

    # Specify the two colors for plotting
    all_c, bad_c = mid_grey, light_grey

    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.bar(x=np.arange(len(props.keys())), height=plt_all_id_fracs,
            yerr=all_id_frac_errs,
            label='All Test Samples', color=all_c, width=0.25, edgecolor='k',
            capsize=7)
    plt.bar(x=np.arange(len(bad_id_fracs)) + 0.3, height=plt_bad_id_fracs,
            yerr=bad_id_frac_errs,
            capsize=7, label='Incorrect Prediction Set', color=bad_c,
            width=0.25, edgecolor='k')
    plt.legend(fontsize=18)
    plt.xticks(np.arange(len(props.keys())) + 0.15, xtick_strs,
               rotation='45')
    plt.ylabel('Fraction of Samples', fontsize=20)

    text_xs = np.arange(len(bad_id_fracs))
    for ii in range(len(bad_id_fracs)):
        plt.text(text_xs[ii] + 0.15, 0.01,
                 '%s' % unique_adi_ids[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__OUT_DIR, 'adi_id_fracs.png'),
                dpi=600, transparent=False)


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load metadata lists of correct and incorrect predictions
    cor_preds = load_pickle(os.path.join(__DATA_DIR, 'byadi_cor_preds.pickle'))
    incor_preds = load_pickle(os.path.join(__DATA_DIR,
                                           'byadi_incor_preds.pickle'))

    plt_adi_id_distros(wrong_preds=incor_preds,
                       all_preds=np.concatenate((cor_preds, incor_preds)))
