"""
Tyson Reimer
October 08th, 2020
"""

import os
import numpy as np
import pandas as pd

import statsmodels.api as sm

from scipy.stats import norm

from umbms import get_proj_path, get_script_logger
from umbms.loadsave import load_pickle

###############################################################################

__DATA_DIR = os.path.join(get_proj_path(), 'output/by-adi-preds/')

phant_info_dir = os.path.join(get_proj_path(), 'data/phant-info/')

###############################################################################


if __name__ == "__main__":

    logger = get_script_logger(__file__)

    # Load metadata lists of correct and incorrect predictions
    cor_preds = load_pickle(os.path.join(__DATA_DIR, 'byadi_cor_preds.pickle'))
    incor_preds = load_pickle(os.path.join(__DATA_DIR,
                                           'byadi_incor_preds.pickle'))

    # Define list of metadata dicts for all predictions
    all_preds = cor_preds + incor_preds

    # Define array indicating correct vs incorrect prediction
    pred_labels = np.zeros([len(all_preds), ])
    pred_labels[:len(cor_preds)] = 1

    # Load phantom info
    phant_info = np.genfromtxt(os.path.join(phant_info_dir, 'phant_info.csv'),
                               delimiter=',',
                               dtype=['<U20', '<U20', float, float, float])

    # All phantom IDs
    phant_ids = np.array(['%s%s' % (ii[0], ii[1]) for ii in phant_info])

    # Init dicts for phantom density and breast volume
    phant_densities = dict()
    phant_vols = dict()
    for ii in range(len(phant_ids)):

        # Store the fibroglandular % by volume
        phant_densities[phant_ids[ii]] = 100 * phant_info[ii][2]

        # Store the adipose volume in cubic cm
        phant_vols[phant_ids[ii]] = phant_info[ii][3] / (10 * 10 * 10)

    tum_presence = np.array([~np.isnan(md['tum_rad']) for md in all_preds])

    tum_preds = np.array(all_preds)[tum_presence]
    tum_labels = pred_labels[tum_presence]

    healthy_preds = np.array(all_preds)[~tum_presence]
    healthy_labels = pred_labels[~tum_presence]

    ###########################################################################

    logger.info('TUMOUR PREDICTIONS')

    # Init metadata dataframe
    md_df = pd.DataFrame()

    # Get the fibroglandular polar radii
    fib_polar_rad = np.array([np.sqrt((md['fib_x'] - md['adi_x']) ** 2
                                      + (md['fib_y'] - md['adi_y']) ** 2)
                              for md in tum_preds])
    md_df['fib_polar_rad'] = fib_polar_rad

    # Get the adipose polar radii
    adi_polar_rad = np.array([np.sqrt(md['adi_x'] ** 2 + md['adi_y'] ** 2)
                              for md in tum_preds])
    md_df['adi_polar_rad'] = adi_polar_rad

    # Get breast density in % by volume from each scan,
    # include in dataframe
    density = np.array([phant_densities[md['phant_id']] for md in tum_preds])
    md_df['density'] = density

    # Get Adipose ID from each scan, include in dataframe
    adi_vols = np.array([phant_vols[md['phant_id']] for md in tum_preds])
    md_df['adi_vol'] = adi_vols

    # Get the tumor radii from each scan, include in dataframe
    tum_rads = np.array([md['tum_rad'] for md in tum_preds])
    tum_rads[np.isnan(tum_rads)] = 0
    md_df['tum_rad'] = tum_rads

    # Get tumor polar radii from each scan, include in dataframe
    tum_polar_rad = np.array([np.sqrt((md['tum_x'] - md['adi_x']) ** 2
                                      + (md['tum_y'] - md['adi_y']) ** 2)
                              for md in tum_preds])
    tum_polar_rad[np.isnan(tum_polar_rad)] = 0
    md_df['tum_polar_rad'] = tum_polar_rad

    # Include tumour z-position in metadata
    tum_zs = np.array([md['tum_z'] for md in tum_preds])
    tum_zs[np.isnan(tum_zs)] = 0
    tum_zs = np.abs(tum_zs)
    tum_zs = np.max(tum_zs) - tum_zs

    # Convert so that it is the distance from the antenna z-plane
    md_df['tum_z'] = tum_zs

    tum_in_fib = np.array([(md['tum_in_fib']) for md in tum_preds])
    md_df['tum_in_fib'] = tum_in_fib

    # Store prediction score in dataframe
    md_df['pred_score'] = tum_labels

    # Create logistic regression model
    model = sm.GLM.from_formula("pred_score ~  "
                                " adi_vol "
                                " + density"
                                " + fib_polar_rad"
                                " + adi_polar_rad"
                                " + tum_rad"
                                " + tum_polar_rad"
                                " + tum_z"
                                " + C(tum_in_fib)"
                                ,
                                family=sm.families.Binomial(),
                                data=md_df)
    results = model.fit()

    # Report results
    logger.info(results.summary2())
    logger.info('\tp-values:')
    logger.info('\t\t%s' % results.pvalues)

    # Critical value - look at 95% confidence intervals
    zstar = norm.ppf(0.95)

    # Report odds ratio and significance level results
    for ii in results.params.keys():

        logger.info('\t%s' % ii)  # Print metadata info

        coeff = results.params[ii]
        std_err = results.bse[ii]

        odds_ratio = np.exp(coeff)  # Get odds ratio

        # Get 95% C.I. for odds ratio
        or_low = np.exp(coeff - zstar * std_err)
        or_high = np.exp(coeff + zstar * std_err)

        # Get p-val
        pval = results.pvalues[ii]

        logger.info('\t\tOdds ratio:\t\t\t%.3e\t(%.3e,\t%.3e)'
                    % (odds_ratio, or_low, or_high))
        logger.info('\t\tp-value:\t\t\t%.3e' % pval)

    ###########################################################################

    print('\n' * 5)
    logger.info('HEALTHY PREDICTIONS')

    # Init metadata dataframe
    md_df = pd.DataFrame()

    # Get the fibroglandular polar radii
    fib_polar_rad = np.array([np.sqrt((md['fib_x'] - md['adi_x']) ** 2
                                      + (md['fib_y'] - md['adi_y']) ** 2)
                              for md in healthy_preds])
    md_df['fib_polar_rad'] = fib_polar_rad

    # Get the adipose polar radii
    adi_polar_rad = np.array([np.sqrt(md['adi_x'] ** 2 + md['adi_y'] ** 2)
                              for md in healthy_preds])
    md_df['adi_polar_rad'] = adi_polar_rad

    # Get breast density in % by volume from each scan,
    # include in dataframe
    density = np.array([phant_densities[md['phant_id']]
                        for md in healthy_preds])
    md_df['density'] = density

    # Get Adipose ID from each scan, include in dataframe
    adi_vols = np.array([phant_vols[md['phant_id']] for md in healthy_preds])
    md_df['adi_vol'] = adi_vols

    # Store prediction score in dataframe
    md_df['pred_score'] = healthy_labels

    # Create logistic regression model
    model = sm.GLM.from_formula("pred_score ~  "
                                " adi_vol "
                                " + density"
                                " + fib_polar_rad"
                                " + adi_polar_rad"
                                ,
                                family=sm.families.Binomial(),
                                data=md_df)
    results = model.fit()

    # Report results
    logger.info(results.summary2())
    logger.info('\tp-values:')
    logger.info('\t\t%s' % results.pvalues)

    # Critical value - look at 95% confidence intervals
    zstar = norm.ppf(0.95)

    # Report odds ratio and significance level results
    for ii in results.params.keys():

        logger.info('\t%s' % ii)  # Print metadata info

        coeff = results.params[ii]
        std_err = results.bse[ii]

        odds_ratio = np.exp(coeff)  # Get odds ratio

        # Get 95% C.I. for odds ratio
        or_low = np.exp(coeff - zstar * std_err)
        or_high = np.exp(coeff + zstar * std_err)

        # Get p-val
        pval = results.pvalues[ii]

        logger.info('\t\tOdds ratio:\t\t\t%.3e\t(%.3e,\t%.3e)'
                    % (odds_ratio, or_low, or_high))
        logger.info('\t\tp-value:\t\t\t%.3e' % pval)
