"""
Tyson Reimer
University of Manitoba
October 15th, 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path
from umbms.loadsave import load_pickle

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'output/roc-figs/')

__OUT_DIR = os.path.join(get_proj_path(), 'output/figs/')
verify_path(__OUT_DIR)

###############################################################################


if __name__ == "__main__":

    # Load ROC data obtained from testing on G1 after training on G1
    cnn_roc = load_pickle(os.path.join(__DATA_DIR, 'cnn_run_8_roc.pickle'))
    dnn_roc = load_pickle(os.path.join(__DATA_DIR, 'dnn_run_15_roc.pickle'))
    logreg_roc = load_pickle(os.path.join(__DATA_DIR,
                                          'logreg_run_0_roc.pickle'))

    # Define colours for the figure
    mid_grey = [118, 113, 113]
    mid_grey = [ii / 255 for ii in mid_grey]

    light_grey = [200, 200, 200]
    light_grey = [ii / 255 for ii in light_grey]

    # Make the figure
    plt.figure(figsize=(10, 10))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=24)
    plt.plot(cnn_roc[0], cnn_roc[1], 'k-', label='CNN')
    plt.plot(dnn_roc[0], dnn_roc[1], color=mid_grey, label='DNN')
    plt.plot(logreg_roc[0], logreg_roc[1], '-', color=light_grey,
             label='Logistic Regression')
    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100),
             'r--', label='Random Classification')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=24)
    plt.xlabel('False Positive Rate', fontsize=28)
    plt.ylabel('True Positive Rate', fontsize=28)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.show()

    plt.savefig(os.path.join(__OUT_DIR, 'roc_curve_fig.png'),
                dpi=600, bbox_inches='tight')
