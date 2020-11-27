"""
Tyson Reimer
University of Manitoba
August 06th, 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from umbms import get_proj_path, get_script_logger, verify_path

from umbms.loadsave import load_pickle, save_pickle

from umbms.ai.augment import full_aug
from umbms.ai.gencompare import correct_g1_ini_ant_ang
from umbms.ai.makesets import get_class_labels
from umbms.ai.preproc import resize_features_for_logreg, to_td
from umbms.ai.metrics import (get_acc, get_sens, get_spec, get_opt_thresh,
                              report_metrics)

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/')

###############################################################################

# Number of epochs to train over
__REG_PARAM = 1e6

###############################################################################

def plt_roc_curve(preds, labels, save_str='', save=False):
    """Plots the ROC curve of the classifier

    Parameters
    ----------
    preds : array_like
        Classifier predictions
    labels : array_like
        True class labels
    save_str : str
        String to use to save fig and data, if save. Should not have
        file extension, should not be full path - just name of .pickle
        and .png files that will be saved
    save : bool
        If True, will save the fig and data
    """

    # Thresholds to use for plt
    thresholds = np.linspace(0, 1, 1000)

    # Init arrays for storing FPR and TPR
    fprs = np.zeros_like(thresholds)
    tprs = np.zeros_like(thresholds)

    for ii in range(np.size(thresholds)):

        # Get TPR here
        tprs[ii] = get_sens(preds=preds, labels=labels,
                            threshold=thresholds[ii])

        # Get FPR here
        fprs[ii] = 1 - get_spec(preds=preds, labels=labels,
                                threshold=thresholds[ii])

    # Make fig
    plt.figure(figsize=(12, 6))
    plt.rc("font", family="Times New Roman")
    plt.tick_params(labelsize=20)
    plt.plot(fprs, tprs, 'k-')
    plt.plot(np.linspace(0, 1, 1000), np.linspace(0, 1, 1000), 'b--')
    plt.xlabel('False Positive Rate', fontsize=24)
    plt.ylabel('True Positive Rate', fontsize=24)
    plt.tight_layout()

    if save:  # If saving

        verify_path(os.path.join(get_proj_path(), 'output/roc-figs/'))
        out_path = os.path.join(get_proj_path(), 'output/roc-figs/')

        plt.savefig(os.path.join(out_path, '%s.png' % save_str), dpi=150)
        plt.close()
        save_pickle(np.array([fprs, tprs]),
                    os.path.join(out_path, '%s.pickle' % save_str))


###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the training data and metadata from Gen-2
    g2_d = load_pickle(os.path.join(__DATA_DIR, 'g2/g2_fd.pickle'))
    g2_md = load_pickle(os.path.join(__DATA_DIR, 'g2/g2_metadata.pickle'))

    # Load the training data and metadata from Gen-1
    g1_d = load_pickle(os.path.join(__DATA_DIR,
                                    'g1-train-test/test_fd.pickle'))
    g1_md = load_pickle(os.path.join(__DATA_DIR,
                                     'g1-train-test/test_md.pickle'))

    # Convert data to time domain, take magnitude, apply window
    g1_d = correct_g1_ini_ant_ang(g1_d)
    g1_d = np.abs(to_td(g1_d))
    g2_d = np.abs(to_td(g2_d))

    # Perform data augmentation
    g2_d, g2_md = full_aug(g2_d, g2_md)

    g2_d = resize_features_for_logreg(g2_d)
    g1_d = resize_features_for_logreg(g1_d)
    g2_labels = get_class_labels(g2_md)
    g1_labels = get_class_labels(g1_md)

    n_runs = 20

    # Init arrays for storing performance metrics
    auc_scores = np.zeros([n_runs, ])
    accs = np.zeros([n_runs, ])
    sens = np.zeros([n_runs, ])
    spec = np.zeros([n_runs, ])

    for run_idx in range(n_runs):

        logger.info('\tWorking on run [%d / %d]...' % (run_idx + 1, n_runs))

        # Define the Logistic Regression model
        model = LogisticRegression(C=__REG_PARAM,
                                   solver='lbfgs',
                                   max_iter=1000)

        # Train the model
        model_hist = model.fit(X=g2_d, y=g2_labels)

        # Calculate the predictions
        g1_preds = model.predict_proba(X=g1_d)

        # Get and store ROC AUC
        g1_auc = 100 * roc_auc_score(y_true=g1_labels, y_score=g1_preds[:, 1])
        auc_scores[run_idx] = g1_auc

        # Get optimal decision threshold
        opt_thresh = get_opt_thresh(preds=g1_preds[:, 1],
                                    labels=g1_labels)

        # Store performance metrics
        accs[run_idx] = 100 * get_acc(preds=g1_preds[:, 1],
                                      labels=g1_labels,
                                      threshold=opt_thresh)
        sens[run_idx] = 100 * get_sens(preds=g1_preds[:, 1],
                                       labels=g1_labels,
                                       threshold=opt_thresh)
        spec[run_idx] = 100 * get_spec(preds=g1_preds[:, 1],
                                       labels=g1_labels,
                                       threshold=opt_thresh)
        # Plot ROC curve
        plt_roc_curve(preds=g1_preds[:, 1], labels=g1_labels,
                      save_str='logreg_run_%d_roc' % run_idx, save=True)

        # Report AUC at this run
        logger.info('\t\tAUC:\t%.2f' % g1_auc)

        # Get the class predictions
        class_preds = g1_preds * np.zeros_like(g1_preds)
        class_preds[g1_preds >= opt_thresh] = 1

    # Report performance metrics to logger
    logger.info('Average performance metrics')
    logger.info('')
    report_metrics(aucs=auc_scores, accs=accs, sens=sens, spec=spec,
                   logger=logger)
