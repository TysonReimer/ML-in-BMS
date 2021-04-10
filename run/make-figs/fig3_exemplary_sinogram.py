"""
Tyson Reimer
University of Manitoba
May 01, 2020
"""

import os
import numpy as np

from umbms import get_proj_path, verify_path
from umbms.loadsave import load_pickle
from umbms.plot.sinogramplot import plot_sino
from umbms.ai.preproc import to_td

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'data/umbmid/g2/')

__FIG_OUT_DIR = os.path.join(get_proj_path(), 'output/figs/')
verify_path(__FIG_OUT_DIR)

###############################################################################


if __name__ == "__main__":

    # Load phantom dataset (UM-BMID Gen1)
    phant_data = load_pickle(os.path.join(__DATA_DIR, 'g2_fd.pickle'))

    # Convert dataset to the time-domain
    sinograms = to_td(phant_data)

    # Take sample sinogram
    sinogram = sinograms[201, :, :]

    # Define scan times
    scan_ts = np.linspace(0.5e-9, 5.5e-9, 35)

    plot_sino(td_data=sinogram, ini_t=scan_ts[0], fin_t=scan_ts[-1],
              save_fig=True,
              save_str=os.path.join(__FIG_OUT_DIR, 'sino_example'),
              transparent=False, dpi=300,
              cbar_fmt='%.3f')
