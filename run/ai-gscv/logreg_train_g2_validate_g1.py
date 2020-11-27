"""
Tyson Reimer
University of Manitoba
February 22nd, 2020
"""

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from umbms import get_proj_path, get_script_logger, verify_path

from umbms.loadsave import load_pickle
from umbms.ai.makesets import get_class_labels
from umbms.ai.augment import full_aug
from umbms.ai.preproc import resize_features_for_logreg, to_td
from umbms.ai.gencompare import correct_g1_ini_ant_ang

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/')

__FIG_OUT_DIR = os.path.join(get_proj_path(), 'output/2006-g2-v-g1/')
verify_path(__FIG_OUT_DIR)

###############################################################################

# Maximum number of iterations to use
__MAX_ITER = 1000

# Number of runs to use to identify best-performing regularization
# parameter
__N_RUNS = 5

# Create array of regularization parameter options over which to do GSCV
__REG_PARAMS = np.logspace(-5, 6, num=11)

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the training data and metadata from Gen-2
    g2_tr_d = load_pickle(os.path.join(__DATA_DIR, 'g2/g2_fd.pickle'))
    g2_tr_md = load_pickle(os.path.join(__DATA_DIR, 'g2/g2_metadata.pickle'))

    # Set the G2 data to be the training data
    train_data = g2_tr_d
    train_md = g2_tr_md

    # Load the G1 validation data
    g1_tr_d = load_pickle(os.path.join(__DATA_DIR,
                                       'g1-train-test/train_fd.pickle'))
    g1_tr_md = load_pickle(os.path.join(__DATA_DIR,
                                        'g1-train-test/train_md.pickle'))

    # Set the G1 data to be the validation data
    val_data = g1_tr_d
    val_md = g1_tr_md

    # Correct the initial antenna position used in G1 scans
    val_data = correct_g1_ini_ant_ang(val_data)

    # Preprocess data, take magnitude and apply time-window, augment
    # training dataset
    val_data = np.abs(to_td(val_data))
    val_data = resize_features_for_logreg(val_data)
    train_data = np.abs(to_td(train_data))
    train_data, train_md = full_aug(train_data, train_md)
    train_data = resize_features_for_logreg(train_data)

    # Get the validation and train set class labels and make categorical
    val_labels = get_class_labels(val_md)
    train_labels = get_class_labels(train_md)

    # Create arrays for storing the AUC on the train and validation
    # sets for this regularization parameter after training with
    # correct labels
    train_set_aucs = np.zeros([__N_RUNS, len(__REG_PARAMS)])
    val_set_aucs = np.zeros([__N_RUNS, len(__REG_PARAMS)])

    for run_idx in range(__N_RUNS):  # For each CV-fold

        logger.info('\tWorking on run [%2d / %2d]...'
                    % (run_idx + 1, __N_RUNS))

        # Iterate over the grid of regularization parameter values
        for ii in range(len(__REG_PARAMS)):

            logger.info('\t\tReg Param Option: [%2d / %2d]'
                        % (ii + 1, len(__REG_PARAMS)))

            # Define the Logistic Regression model
            model = LogisticRegression(C=__REG_PARAMS[ii],
                                       solver='lbfgs',
                                       max_iter=__MAX_ITER)

            # Train the DNN
            model_hist = model.fit(X=train_data, y=train_labels)

            # Get the predictions on train/val sets after training
            train_preds = model.predict_proba(X=train_data)[:, 1]
            val_preds = model.predict_proba(X=val_data)[:, 1]

            # Get AUC on train/val sets
            train_auc = roc_auc_score(y_true=train_labels, y_score=train_preds)
            val_auc = roc_auc_score(y_true=val_labels, y_score=val_preds)

            # Store AUCs and loss at each epoch for train set
            train_set_aucs[run_idx, ii] = 100 * train_auc

            # Store AUCs and loss at each epoch for val set
            val_set_aucs[run_idx, ii] = 100 * val_auc

    ###########################################################################
    # Plot results

    viridis = get_cmap('viridis')

    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.plot(__REG_PARAMS, np.mean(train_set_aucs, axis=0), 'o',
             color=viridis(0), label='Train Set (Gen-2)')
    plt.fill_between(x=__REG_PARAMS,
                     y1=np.mean(train_set_aucs, axis=0)
                        - np.std(train_set_aucs, axis=0),
                     y2=np.mean(train_set_aucs, axis=0)
                        + np.std(train_set_aucs, axis=0),
                     color=viridis(0),
                     alpha=0.30)
    plt.plot(__REG_PARAMS, np.mean(val_set_aucs, axis=0), 'o',
             color=viridis(0.7), label='Validation Set (Gen-1')
    plt.fill_between(x=__REG_PARAMS,
                     y1=np.mean(val_set_aucs, axis=0)
                        - np.std(val_set_aucs, axis=0),
                     y2=np.mean(val_set_aucs, axis=0)
                        + np.std(val_set_aucs, axis=0),
                     color=viridis(0.7),
                     alpha=0.30)
    val_max_auc = np.max(np.mean(val_set_aucs, axis=0))
    plt.plot(__REG_PARAMS, val_max_auc * np.ones_like(__REG_PARAMS),
             color='k', linestyle='--', label='Max Validation AUC')
    plt.plot(__REG_PARAMS, 50 * np.ones_like(__REG_PARAMS),
             color='r', linestyle='--', label='Random Classification')
    plt.plot(__REG_PARAMS, 100 * np.ones_like(__REG_PARAMS),
             color=viridis(0.95), linestyle='--',
             label='Perfect Classification')
    plt.xscale('log')
    plt.legend(fontsize=20, loc='right')
    plt.xlabel('Number of Epochs', fontsize=22)
    plt.ylabel('Average AUC (%)', fontsize=22)
    plt.ylim([45, 102.5])
    plt.title('Logistic Regression Performance on Training (Gen-2) and '
              'Validation (Gen-1) Sets',
              fontsize=26)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__FIG_OUT_DIR,
                             'logreg_train_g2_validate_g1_aucs.png'),
                dpi=600, transparent=True)
    plt.savefig(
        os.path.join(__FIG_OUT_DIR,
                     'logreg_train_g2_validate_g1_aucs_not_transparent.png'),
        dpi=600, transparent=False)
