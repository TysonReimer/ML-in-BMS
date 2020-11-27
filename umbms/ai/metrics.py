"""
Tyson Reimer
University of Manitoba
February 7th, 2020
"""

import numpy as np

###############################################################################


def _get_tp_tn_fp_fn(preds, labels, threshold=0.5):
    """Gets the true positives/negatives and false positives/negatives

    Parameters
    ----------
    preds : array_like
        The class predictions from the model
    labels : array_like
        The binary class labels
    threshold : float
        The decision threshold

    Returns
    -------
    tp : int
        The number of true positive predictions
    tn : int
        The number of true negative predictions
    fp : int
        The number of false positive predictions
    fn : int
        The number of false negative predictions
    """

    tp = np.sum(np.logical_and(preds >= threshold, labels == 1))
    tn = np.sum(np.logical_and(preds <= threshold, labels == 0))
    fp = np.sum(np.logical_and(preds >= threshold, labels == 0))
    fn = np.sum(np.logical_and(preds <= threshold, labels == 1))

    return tp, tn, fp, fn


def get_acc(preds, labels, threshold=0.5):
    """Gets the accuracy score of the predictions

    Parameters
    ----------
    preds : array_like
        Class predictions for each sample
    labels : array_like
        Binary class labels of each sample
    threshold : float
        The decision threshold

    Returns
    -------
    acc : float
        The model accuracy
    """

    tp, tn, fp, fn = _get_tp_tn_fp_fn(preds=preds, labels=labels,
                                      threshold=threshold)

    acc = (tp + tn) / (tp + tn + fn + fp)

    return acc


def get_sens(preds, labels, threshold=0.5):
    """Gets the sensitivity score of the predictions

    Parameters
    ----------
    preds : array_like
        Class predictions for each sample
    labels : array_like
        Binary class labels of each sample
    threshold : float
        The decision threshold

    Returns
    -------
    sens : float
        The model sensitivity
    """

    tp, _, _, fn = _get_tp_tn_fp_fn(preds=preds, labels=labels,
                                    threshold=threshold)

    sens = tp / (tp + fn)

    return sens


def get_spec(preds, labels, threshold=0.5):
    """Gets the specificity score of the predictions

    Parameters
    ----------
    preds : array_like
        Class predictions for each sample
    labels : array_like
        Binary class labels of each sample
    threshold : float
        The decision threshold

    Returns
    -------
    spec : float
        The model specificity
    """

    _, tn, fp, _ = _get_tp_tn_fp_fn(preds=preds, labels=labels,
                                    threshold=threshold)

    spec = tn / (tn + fp)

    return spec


def get_opt_thresh(preds, labels, n_thresholds=10000):
    """Get the optimum threshold value to maximize sens and spec

    Parameters
    ----------
    preds : array_like
        Class predictions for each sample
    labels : array_like
        Binary class labels of each sample
    n_thresholds : int
        Number of thresholds between 0 and 1 that will be examined

    Returns
    -------
    opt_thresh : float
        The decision threshold that maximizes the euclidean norm
        of sensitivity and specificity, equivalent to finding
        the point on the ROC curve that is closest to the point
        (0, 1) on that plot
    """

    thresholds = np.linspace(0, 1, n_thresholds)  # Thresholds to search
    best_score = -1  # Init best score
    opt_thresh = -1  # Init optimal threshold

    for thresh in thresholds:  # For each threshold

        # Get the score here
        score_here = get_acc(preds=preds, labels=labels, threshold=thresh)
        if score_here > best_score:  # If this is the best score yet

            best_score = score_here  # Set it to be the best score
            opt_thresh = thresh  # Set the optimal threshold to be this

    return opt_thresh


def report_metrics(aucs, accs, sens, spec, logger):
    """Reports metric averages and stdevs to logger

    Parameters
    ----------
    aucs : array_like
        Array of ROC AUC over some runs
    accs : array_like
        Array of accuracies over some runs
    sens : array_like
        Array of sensitivities over some runs
    spec : array_like
        Array of specificities over some runs
    logger :
        logging object
    """

    logger.info('\t| %14s | %14s | %14s | %14s |'
                % ('AUC', 'Acc', 'Sens', 'Spec'))
    logger.info('\t| %.2f +/- %.2f | %.2f +/- %.2f | %.2f +/- %.2f |'
                '%.2f +/- %.2f |'
                % (np.mean(aucs), np.std(aucs), np.mean(accs), np.std(accs),
                   np.mean(sens), np.std(sens), np.mean(spec), np.std(spec)))
