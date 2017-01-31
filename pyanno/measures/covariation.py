# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Standard reliability and covariation measures."""



import numpy as np
import scipy.stats
from pyanno.measures.helpers import all_invalid

from pyanno.util import is_valid, MISSING_VALUE

import logging
logger = logging.getLogger(__name__)

def pearsons_rho(annotations1, annotations2, nclasses=None):
    """Compute Pearson's product-moment correlation coefficient.

    See also :func:`~pyanno.measures.helpers.pairwise_matrix`.

    **References:**

    * `Wikipedia entry
      <http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient>`_

    Arguments
    ---------
    annotations1 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    annotations2 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    nclasses : int
        Number of annotation classes. If None, `nclasses` is inferred from the
        values in the annotations

    Returns
    -------
    stat : float
        The value of the statistics
    """

    valid = is_valid(annotations1) & is_valid(annotations2)
    if all(~valid):
        logger.debug('No valid annotations')
        return np.nan

    rho, pval = scipy.stats.pearsonr(annotations1[valid], annotations2[valid])
    return rho


def spearmans_rho(annotations1, annotations2, nclasses=None):
    """Compute Spearman's rank correlation coefficient.

    See also :func:`~pyanno.measures.helpers.pairwise_matrix`.

    **References:**

    * `Wikipedia entry
      <http://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient>`_

    Arguments
    ---------
    annotations1 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    annotations2 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    nclasses : int
        Number of annotation classes. If None, `nclasses` is inferred from the
        values in the annotations

    Returns
    -------
    stat : float
        The value of the statistics
    """

    valid = is_valid(annotations1) & is_valid(annotations2)
    if all(~valid):
        logger.debug('No valid annotations')
        return np.nan

    rho, pval = scipy.stats.spearmanr(annotations1[valid], annotations2[valid])
    return rho


def cronbachs_alpha(annotations, nclasses=None):
    """Compute Cronbach's alpha.

    **References:**

    * Cronbach, L. J. (1951). "Coefficient alpha and the internal structure
      of tests." Psychometrika, 16(3), 297-334.

    * `Wikipedia entry
      <http://en.wikipedia.org/wiki/Cronbach%27s_alpha>`_

    Arguments
    ---------
    annotations : ndarray, shape = (n_items, n_annotators)
        Array of annotations for multiple annotators. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    nclasses : int
        Number of annotation classes. If None, `nclasses` is inferred from the
        values in the annotations

    Returns
    -------
    stat : float
        The value of the statistics
    """

    if all_invalid(annotations):
        logger.debug('No valid annotations')
        return np.nan

    nitems = annotations.shape[0]
    valid_anno = np.ma.masked_equal(annotations, MISSING_VALUE)

    item_var = valid_anno.var(1, ddof=1)
    total_var = valid_anno.sum(0).var(ddof=1)

    return nitems/(nitems - 1.) * (1. - item_var.sum() / total_var)
