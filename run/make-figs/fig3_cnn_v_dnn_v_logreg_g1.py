"""
Tyson Reimer
University of Manitoba
May 04, 2020
"""

import os
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path

###############################################################################

__SAVE_FIG = True

__OUT_DIR = os.path.join(get_proj_path(), 'output/figs/')
verify_path(__OUT_DIR)

###############################################################################

# Define dark grey color
dark_grey = [38, 38, 38]
dark_grey = [ii / 255 for ii in dark_grey]

# Define mid grey color
mid_grey = [118, 113, 113]
mid_grey = [ii / 255 for ii in mid_grey]

# Define light  grey color
light_grey = [217, 217, 217]
light_grey = [ii / 255 for ii in light_grey]

###############################################################################

# Define LogReg metrics in format:
# [auc, auc_std, acc, acc_std, sens, sens_std, spec, spec_std]
logreg_metrics = [43.8, 53.2, 9.7, 96.8]

# Define CNN metrics in same format
cnn_metrics = [78, 3, 75, 3, 82, 9, 70, 10]

dnn_metrics = [50, 8, 56, 4, 30, 20, 80, 20]

###############################################################################


if __name__ == "__main__":

    cnn_xs = [1, 4.5, 8, 11.5]
    dnn_xs = [2, 5.5, 9, 12.5]
    logreg_xs = [3, 6.5, 10, 13.5]

    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=20)

    plt.bar(cnn_xs,
            height=cnn_metrics[::2],
            yerr=cnn_metrics[1::2],
            width=0.75,
            capsize=10,
            linewidth=1,
            color=dark_grey,
            edgecolor='k',
            label='CNN')

    plt.bar(dnn_xs,
            height=dnn_metrics[::2],
            yerr=dnn_metrics[1::2],
            width=0.75,
            capsize=10,
            linewidth=1,
            color=mid_grey,
            edgecolor='k',
            label='DNN')

    plt.bar(logreg_xs,
            height=logreg_metrics,
            width=0.75,
            capsize=10,
            linewidth=1,
            color=light_grey,
            edgecolor='k',
            label='Logistic Regression')

    plt.legend(fontsize=18,
               loc='lower left',
               framealpha=0.95)

    plt.xticks([2, 5.5, 9, 12.5],
               ['ROC AUC', 'Accuracy', 'Sensitivity', 'Specificity'],
               size=20)

    plt.ylabel('Metric Value (%)', fontsize=22)

    cnn_text_ys = [70, 69, 66, 47]
    dnn_text_ys = [35, 43, 24, 56]
    lr_text_ys = [29, 35, 5, 90]

    for ii in range(len(cnn_xs)):
        plt.text(cnn_xs[ii], cnn_text_ys[ii],
                 '(%s ' % cnn_metrics[ii * 2] + r'$\mathdefault{\pm}$'
                 + ' %s)%%' % cnn_metrics[ii * 2 + 1],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(dnn_xs)):
        plt.text(dnn_xs[ii], dnn_text_ys[ii],
                 '(%s ' % dnn_metrics[ii * 2] + r'$\mathdefault{\pm}$'
                 + ' %s)%%' % dnn_metrics[ii * 2 + 1],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    for ii in range(len(logreg_xs)):

        plt.text(logreg_xs[ii], lr_text_ys[ii],
                 '%s%%' % logreg_metrics[ii],
                 size=16,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9})

    # Set appropriate y-limit
    plt.ylim([0, 105])
    plt.tight_layout()  # Make everything fit nicely
    plt.show()  # Display the plot

    if __SAVE_FIG:
        plt.savefig(os.path.join(__OUT_DIR, 'cnn_v_dnn_v_logreg.png'),
                    transparent=False, dpi=300)
