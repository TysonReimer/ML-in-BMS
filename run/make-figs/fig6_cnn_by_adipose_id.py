"""
Tyson Reimer
University of Manitoba
May 04, 2020
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from umbms import get_proj_path, verify_path

###############################################################################

__SAVE_FIG = True

__OUT_DIR = os.path.join(get_proj_path(), 'output/figs/')
verify_path(__OUT_DIR)

###############################################################################

# Define blue to match ppt color
dark_grey = [38, 38, 38]
dark_grey = [ii / 255 for ii in dark_grey]

mid_grey = [118, 113, 113]
mid_grey = [ii / 255 for ii in mid_grey]

# Define grey color that goes well with the blue color
light_grey = [217, 217, 217]
light_grey = [ii / 255 for ii in light_grey]

###############################################################################

adi_id_strs = ['A1', 'A11', 'A12', 'A13', 'A14', 'A2', 'A15', 'A16', 'A3']
old_id_strs = ['A1', 'A2', 'A3', 'A11', 'A12', 'A13', 'A14', 'A15', 'A16']
correct_order = np.array([old_id_strs.index(adi_id_strs[ii])
                          for ii in range(len(adi_id_strs))])

# Define CNN metrics in same format
auc_avgs = np.array([44, 91, 60, 60, 96, 76, 89, 93, 96])
auc_stds = np.array([6, 2, 10, 6, 2, 4, 3, 6, 3])
adi_vols = np.array([294710, 729324, 1113320, 458352, 567850, 652080,
                     713177, 1028750, 1034420]) * 0.001

auc_avgs = auc_avgs[correct_order]
auc_stds = auc_stds[correct_order]
adi_vols = adi_vols[correct_order]

###############################################################################


if __name__ == "__main__":

    vol_xs = [1, 3.5, 6, 8.5, 11, 13.5, 16, 18.5, 21]
    cnn_xs = [2, 4.5, 7, 9.5, 12, 14.5, 17, 19.5, 22]

    label_xs = [1.5, 4, 6.5, 9, 11.5, 14, 16.5, 19, 21.5]

    plt.figure(figsize=(12, 6))
    plt.rc('font', family='Times New Roman')
    plt.tick_params(labelsize=20)
    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.tick_params(labelsize=20)

    vol_bars = ax.bar(vol_xs,
                      height=adi_vols,
                      width=0.75,
                      capsize=7,
                      linewidth=1,
                      color=light_grey,
                      edgecolor='k',
                      label='Adipose Shell Volume',
                      zorder=1)
    ax.set_ylim([0, 1200])
    ax.set_ylabel(r'Adipose Shell Volume (cm$^{\mathdefault{3}}$)',
                   fontsize=22)

    cnn_text_ys = np.array([30, 80, 45, 45, 86, 62, 71, 88, 82])
    cnn_text_ys = cnn_text_ys[correct_order]

    for ii in range(len(cnn_xs)):
        ax2.text(cnn_xs[ii], cnn_text_ys[ii],
                 '(%s ' % auc_avgs[ii] + r'$\mathdefault{\pm}$'
                 + ' %s)%%' % auc_stds[ii],
                 size=20,
                 color='k',
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox={'facecolor': 'w',
                       'alpha': 0.9},
                zorder=3)
    cnn_bars = ax2.bar(cnn_xs,
                       height=auc_avgs,
                       yerr=auc_stds,
                       width=0.75,
                       capsize=7,
                       linewidth=1,
                       color=dark_grey,
                       edgecolor='k',
                       label='CNN Performance',
                       zorder=2)

    bars1, labels1 = ax.get_legend_handles_labels()
    bars2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(bars1 + bars2, labels1 + labels2, loc='lower right', fontsize=20)
    plt.xticks(label_xs, adi_id_strs, size=20)

    plt.xlabel('Adipose Shell ID Assigned to Test Set', fontsize=22)
    ax2.set_ylabel('ROC AUC (%)', fontsize=22)

    # Set appropriate y-limit
    ax2.set_ylim([0, 100])

    plt.tight_layout()  # Make everything fit nicely
    plt.show()  # Display the plot

    if __SAVE_FIG:
        plt.savefig(os.path.join(__OUT_DIR, 'cnn_by_adi_v_adi_vol.png'),
                    transparent=False, dpi=300)
