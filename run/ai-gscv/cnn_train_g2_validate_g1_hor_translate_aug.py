"""
Tyson Reimer
University of Manitoba
February 22nd, 2020
"""

import os
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import clear_session

from umbms import get_proj_path, get_script_logger, verify_path

from umbms.loadsave import load_pickle
from umbms.ai.makesets import get_class_labels
from umbms.ai.models import get_sino_cnn
from umbms.ai.preproc import resize_features_for_keras, to_td
from umbms.ai.augment import aug_hor_translate
from umbms.ai.gencompare import correct_g1_ini_ant_ang

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/')

__FIG_OUT_DIR = os.path.join(get_proj_path(), 'output/2006-g2-v-g1/')
verify_path(__FIG_OUT_DIR)

###############################################################################

# Number of epochs to train over
__N_EPOCHS = 1000

# Set the bounds on the time-domain window that will be applied to all
# data before training or testing
__T_WIN_LOW = 5
__T_WIN_HIGH = 40

# Number of runs to use to identify best-performing number of epochs
__N_RUNS = 10

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
    val_data = resize_features_for_keras(val_data)
    train_data = np.abs(to_td(train_data))
    train_data, train_md = aug_hor_translate(train_data, train_md, step_size=5)
    train_data = resize_features_for_keras(train_data)

    # Get the validation and train set class labels and make categorical
    val_labels = get_class_labels(val_md)
    val_labels = to_categorical(val_labels)
    train_labels = get_class_labels(train_md)
    train_labels = to_categorical(train_labels)

    # Create arrays for storing the AUC on the train and validation
    # sets for this regularization parameter after training with
    # correct labels
    train_set_aucs = np.zeros([__N_RUNS, __N_EPOCHS])
    val_set_aucs = np.zeros([__N_RUNS, __N_EPOCHS])

    train_set_loss = np.zeros([__N_RUNS, __N_EPOCHS])
    val_set_loss = np.zeros([__N_RUNS, __N_EPOCHS])

    for run_idx in range(__N_RUNS):  # For each CV-fold

        logger.info('\tWorking on run [%2d / %2d]...'
                    % (run_idx + 1, __N_RUNS))

        # Define the CNN
        model = get_sino_cnn(input_shape=np.shape(train_data)[1:],
                             lr=0.001)

        # Train the CNN
        model_hist = model.fit(x=train_data, y=train_labels,
                               epochs=__N_EPOCHS, shuffle=True,
                               batch_size=2048, verbose=True,
                               validation_data=(val_data, val_labels))

        # Store AUCs and loss at each epoch for train set
        train_set_aucs[run_idx, :] = \
            (100 * np.array(model_hist.history['auc']))
        train_set_loss[run_idx, :] = np.array(model_hist.history['loss'])

        # Store AUCs and loss at each epoch for val set
        val_set_aucs[run_idx, :] = \
            (100 * np.array(model_hist.history['val_auc']))
        val_set_loss[run_idx, :] = np.array(model_hist.history['val_loss'])

        clear_session()  # Reset network

    ###########################################################################
    # Plot results

    viridis = get_cmap('viridis')

    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=18)
    plt.plot(np.arange(__N_EPOCHS), np.mean(train_set_aucs, axis=0),
             color=viridis(0), label='Train Set (Gen-2)')
    plt.fill_between(x=np.arange(__N_EPOCHS),
                     y1=np.mean(train_set_aucs, axis=0)
                        - np.std(train_set_aucs, axis=0),
                     y2=np.mean(train_set_aucs, axis=0)
                        + np.std(train_set_aucs, axis=0),
                     color=viridis(0),
                     alpha=0.30)
    plt.plot(np.arange(__N_EPOCHS), np.mean(val_set_aucs, axis=0),
             color=viridis(0.7), label='Validation Set (Gen-1')
    plt.fill_between(x=np.arange(__N_EPOCHS),
                     y1=np.mean(val_set_aucs, axis=0)
                        - np.std(val_set_aucs, axis=0),
                     y2=np.mean(val_set_aucs, axis=0)
                        + np.std(val_set_aucs, axis=0),
                     color=viridis(0.7),
                     alpha=0.30)
    val_max_auc = np.max(np.mean(val_set_aucs, axis=0))
    plt.plot(np.arange(__N_EPOCHS), val_max_auc * np.ones([__N_EPOCHS, ]),
             color='k', linestyle='--', label='Max Validation AUC')
    plt.plot(np.arange(__N_EPOCHS), 50 * np.ones([__N_EPOCHS, ]),
             color='r', linestyle='--', label='Random Classification')
    plt.plot(np.arange(__N_EPOCHS), 100 * np.ones([__N_EPOCHS, ]),
             color=viridis(0.95), linestyle='--',
             label='Perfect Classification')
    plt.legend(fontsize=20, loc='right')
    plt.xlabel('Number of Epochs', fontsize=22)
    plt.ylabel('Average AUC (%)', fontsize=22)
    plt.ylim([45, 102.5])
    plt.title('CNN Performance on Training (Gen-2) and '
              'Validation (Gen-1) Sets',
              fontsize=26)
    plt.tight_layout()
    plt.show()
    plt.savefig(os.path.join(__FIG_OUT_DIR,
                             'cnn_train_g2_validate_g1_aucs_'
                             'hor_translate_aug.png'),
                dpi=600, transparent=True)
    plt.savefig(
        os.path.join(__FIG_OUT_DIR,
                     'cnn_train_g2_validate_g1_aucs_hor_translate_aug_'
                     'not_transparent.png'),
        dpi=600, transparent=False)
