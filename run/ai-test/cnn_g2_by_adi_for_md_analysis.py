"""
Tyson Reimer
University of Manitoba
February 7th, 2020
"""

import os
import numpy as np

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session

from umbms import get_proj_path, get_script_logger, verify_path

from umbms.loadsave import load_pickle, save_pickle

from umbms.ai.makesets import get_class_labels
from umbms.ai.models import get_sino_cnn
from umbms.ai.preproc import resize_features_for_keras, to_td
from umbms.ai.augment import full_aug
from umbms.ai.metrics import get_acc, get_sens, get_spec, get_opt_thresh

from sklearn.metrics import roc_auc_score

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/')

###############################################################################

__N_EPOCHS = 496

__N_RUNS = 1

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load the training data and metadata from Gen-2
    g2_d = load_pickle(os.path.join(__DATA_DIR, 'g2/g2_fd.pickle'))
    g2_md = load_pickle(os.path.join(__DATA_DIR, 'g2/g2_metadata.pickle'))
    g2_md = np.array(g2_md)

    # Convert to the time-domain, crop the signals, take magnitude
    g2_d = np.abs(to_td(g2_d))

    # Get all of the Adipose IDs
    adi_ids = np.array([md['phant_id'].split('F')[0] for md in g2_md])
    unique_adi_ids = np.unique(adi_ids)

    adi_results = dict()  # Init dict for storing results

    # Init lists for storing metadata of samples which are
    # incorrectly / correctly classified
    incor_preds = []
    cor_preds = []

    for adi_id in unique_adi_ids:  # For each adipose ID

        logger.info('\tWorking on Adi ID:\t%s' % adi_id)

        # Get the indices of the samples with the target Adi ID
        tar_idxs = np.array([this_adi_id in [adi_id]
                             for this_adi_id in adi_ids])

        # Use the samples with this adipose ID as the test set,
        # all others assigned to train set
        test_data = g2_d[tar_idxs, :, :]
        train_data = g2_d[~tar_idxs, :, :]
        test_md = g2_md[tar_idxs]
        train_md = g2_md[~tar_idxs]

        # Perform data augmentation on the training set here
        train_data, train_md = full_aug(train_data, train_md)

        # Get class labels for train/test sets here
        test_labels = get_class_labels(test_md)
        train_labels = get_class_labels(train_md)
        test_labels = to_categorical(test_labels)
        train_labels = to_categorical(train_labels)

        # Resize data for use with keras
        test_data = resize_features_for_keras(test_data)
        train_data = resize_features_for_keras(train_data)

        # Iterate over number of desired runs
        for run_idx in range(__N_RUNS):

            logger.info('\tWorking on run [%2d / %2d]...'
                        % (run_idx + 1, __N_RUNS))

            # Create the cnn model
            model = get_sino_cnn(input_shape=np.shape(train_data)[1:],
                                 lr=0.001)

            # Train the model
            model.fit(x=train_data, y=train_labels,
                      epochs=__N_EPOCHS, shuffle=True,
                      batch_size=2056, verbose=False)

            # Predict on the test set
            test_preds = model.predict(x=test_data)

            # Get metrics at 0.50 decision threshold
            auc = roc_auc_score(y_true=test_labels, y_score=test_preds)
            acc = get_acc(preds=test_preds[:, 1], labels=test_labels[:, 1])
            sens = get_sens(preds=test_preds[:, 1], labels=test_labels[:, 1])
            spec = get_spec(preds=test_preds[:, 1], labels=test_labels[:, 1])

            # Get metrics at optimal decision threshold
            opt_thresh = get_opt_thresh(preds=test_preds[:, 1],
                                        labels=test_labels[:, 1],
                                        n_thresholds=10000)
            test_acc_opt = get_acc(preds=test_preds[:, 1],
                                   labels=test_labels[:, 1],
                                   threshold=opt_thresh)
            test_sens_opt = get_sens(preds=test_preds[:, 1],
                                     labels=test_labels[:, 1],
                                     threshold=opt_thresh)
            test_spec_opt = get_spec(preds=test_preds[:, 1],
                                     labels=test_labels[:, 1],
                                     threshold=opt_thresh)

            # Get the class predictions
            class_preds = test_preds * np.zeros_like(test_preds)
            class_preds[test_preds >= opt_thresh] = 1

            # Store correct and incorrect predictions
            for s_idx in range(np.size(test_preds, axis=0)):
                if class_preds[s_idx, 1] != test_labels[s_idx, 1]:
                    incor_preds.append(test_md[s_idx])
                else:
                    cor_preds.append(test_md[s_idx])

            # Clear the keras model
            clear_session()


            logger.info('\t\tResults here at 0.5 threshold:')
            logger.info('\t\t\tAUC: %.3f\tAcc: %.3f\tSens: %.3f\tSpec: %.3f'
                        % (100 * auc, 100 * acc, 100 * sens, 100 * spec))
            logger.info('\t\tResults here at optimal threshold:')
            logger.info('\t\t\tAUC: %.3f\tAcc: %.3f\tSens: %.3f\tSpec: %.3f'
                        % (auc * 100, 100 * test_acc_opt, test_sens_opt,
                           test_spec_opt))

    # Save the results to a .pickle file
    save_dir = os.path.join(get_proj_path(), 'output/g2-by-adi//')
    verify_path(save_dir)
    save_pickle(adi_results,
                os.path.join(save_dir, 'g2_by_adi_metrics.pickle'))

    # Save the incorrect predictions
    out_dir = os.path.join(get_proj_path(), 'output/by-adi-preds/')
    verify_path(out_dir)

    save_pickle(incor_preds,
                os.path.join(out_dir, 'byadi_incor_preds.pickle'))
    save_pickle(cor_preds,
                os.path.join(out_dir, 'byadi_cor_preds.pickle'))
