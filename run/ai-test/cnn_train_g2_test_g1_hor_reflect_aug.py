"""
Tyson Reimer
University of Manitoba
August 06th, 2020
"""

import os
import numpy as np

from sklearn.metrics import roc_auc_score

from tensorflow.keras.backend import clear_session
from tensorflow.keras.utils import to_categorical

from umbms import get_proj_path, get_script_logger

from umbms.loadsave import load_pickle

from umbms.ai.augment import aug_hor_ref
from umbms.ai.gencompare import correct_g1_ini_ant_ang
from umbms.ai.models import get_sino_cnn
from umbms.ai.makesets import get_class_labels
from umbms.ai.preproc import resize_features_for_keras, to_td
from umbms.ai.metrics import (get_acc, get_sens, get_spec, get_opt_thresh,
                              report_metrics)

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/')

###############################################################################

# Number of epochs to train over
__N_EPOCHS = 1717

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
    g2_d, g2_md = aug_hor_ref(g2_d, g2_md)

    g2_d = resize_features_for_keras(g2_d)
    g1_d = resize_features_for_keras(g1_d)
    g2_labels = to_categorical(get_class_labels(g2_md))
    g1_labels = to_categorical(get_class_labels(g1_md))

    n_runs = 20

    # Init arrays for storing performance metrics
    auc_scores = np.zeros([n_runs, ])
    accs = np.zeros([n_runs, ])
    sens = np.zeros([n_runs, ])
    spec = np.zeros([n_runs, ])

    for run_idx in range(n_runs):

        logger.info('\tWorking on run [%d / %d]...' % (run_idx + 1, n_runs))

        # Get model
        cnn = get_sino_cnn(input_shape=np.shape(g2_d)[1:], lr=0.001)

        # Train the model
        cnn.fit(x=g2_d, y=g2_labels,
                epochs=__N_EPOCHS,
                shuffle=True,
                batch_size=2048,
                verbose=False)

        # Calculate the predictions
        g1_preds = cnn.predict(x=g1_d)

        # Get and store ROC AUC
        g1_auc = 100 * roc_auc_score(y_true=g1_labels, y_score=g1_preds)
        auc_scores[run_idx] = g1_auc

        # Get optimal decision threshold
        opt_thresh = get_opt_thresh(preds=g1_preds[:, 1],
                                    labels=g1_labels[:, 1])

        # Store performance metrics
        accs[run_idx] = 100 * get_acc(preds=g1_preds[:, 1],
                                      labels=g1_labels[:, 1],
                                      threshold=opt_thresh)
        sens[run_idx] = 100 * get_sens(preds=g1_preds[:, 1],
                                       labels=g1_labels[:, 1],
                                       threshold=opt_thresh)
        spec[run_idx] = 100 * get_spec(preds=g1_preds[:, 1],
                                       labels=g1_labels[:, 1],
                                       threshold=opt_thresh)

        # Report AUC at this run
        logger.info('\t\tAUC:\t%.2f' % g1_auc)

        # Get the class predictions
        class_preds = g1_preds * np.zeros_like(g1_preds)
        class_preds[g1_preds >= opt_thresh] = 1

        # Reset the model
        clear_session()

    # Report performance metrics to logger
    logger.info('Average performance metrics')
    logger.info('')
    report_metrics(aucs=auc_scores, accs=accs, sens=sens, spec=spec,
                   logger=logger)
