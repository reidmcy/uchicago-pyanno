# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)



import numpy as np
from pyanno.util import labels_count, labels_frequency, is_valid


def pairwise_matrix(pairwise_statistic, annotations, *args, **kwargs):
    """Compute the matrix of all combinations of a pairwise statistics.

    This function applies an agreement or covariation statistic that is only
    defined for pairs of annotators to all combinations of annotators pairs,
    and returns a matrix of the result.

    Example ::

        >>> from pyanno.measures import pairwise_matrix, cohens_kappa
        >>> stat_matrix = pairwise_matrix(cohens_kappa, annotations, nclasses=4)

    Arguments
    ---------
    pairwise_statistics : function
        Function accepting as first two arguments two 1D array of
        annotations, and returning a single scalar measuring some annotations
        statistics.

    annotations : ndarray, shape = (n_items, n_annotators)
        Annotations in pyanno format.

    args : any
        Additional arguments passed to `pairwise_statistics`.

    kwargs : any
        Additional keyword arguments passed to `pairwise_statistics`.

    Returns
    -------
    stat_matrix : ndarray, shape = (n_annotators, n_annotators)
        `stat_matrix[i,j]` is the value of `pairwise_statistics` applied to
        the annotations of annotators `i` and `j`
    """

    nannotators = annotations.shape[1]

    pairwise = np.empty((nannotators, nannotators), dtype=float)
    for i in range(nannotators):
        for j in range(nannotators):
            pairwise[i,j] = pairwise_statistic(annotations[:,i],
                                               annotations[:,j],
                                               *args, **kwargs)

    return pairwise


def confusion_matrix(annotations1, annotations2, nclasses):
    """Compute confusion matrix from pairs of annotations.

    **References**

    * `Wikipedia entry <http://en.wikipedia.org/wiki/Confusion_matrix>`_

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
    conf_mat : ndarray, shape = (n_classes, n_classes)
        Confusion matrix; conf_mat[i,j] = number of observations that was
        annotated as category `i` by annotator 1 and as `j` by annotator 2
    """

    conf_mat = np.empty((nclasses, nclasses), dtype=float)
    for i in range(nclasses):
        for j in range(nclasses):
            conf_mat[i, j] = np.sum(np.logical_and(annotations1 == i,
                                                   annotations2 == j))

    return conf_mat


def coincidence_matrix(annotations, nclasses):
    """Build coincidence matrix.

    The element c,k of the coincidence matrix contains the number of c-k pairs
    in the data (across annotators), over the total number of observed pairs.

    **Reference**

    * `Wikipedia entry
      <http://en.wikipedia.org/wiki/Krippendorff%27s_Alpha#Coincidence_matrices>`_

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
    coinc_mat : ndarray, shape = (n_classes, n_classes)
        Coincidence matrix
    """

    # total number of annotations in row
    nannotations = is_valid(annotations).sum(1).astype(float)
    valid = nannotations > 1

    nannotations = nannotations[valid]
    annotations = annotations[valid,:]

    # number of annotations of class c in row
    nc_in_row = np.empty((nannotations.shape[0], nclasses), dtype=int)
    for c in range(nclasses):
        nc_in_row[:, c] = (annotations == c).sum(1)

    coincidences = np.empty((nclasses, nclasses), dtype=float)
    for c in range(nclasses):
        for k in range(nclasses):
            if c==k:
                nck_pairs = nc_in_row[:, c] * (nc_in_row[:, c] - 1)
            else:
                nck_pairs = nc_in_row[:, c] * nc_in_row[:, k]
            coincidences[c, k] = (nck_pairs / (nannotations - 1.)).sum()

    return coincidences


def chance_adjusted_agreement(observed_agreement, chance_agreement):
    """Return the chance-adjusted agreement given the specified agreement
    and expected agreement.

    Defined by (observed_agreement - chance_agreement)/(1.0 - chance_agreement)

    Arguments
    ---------
    observed_agreement : float
        Agreement computed from the data

    chance_agreement : float
        Agreement expected by chance give the assumptions of the statistics

    Return
    ------
    result : float
        Chance adjusted agreement value
    """

    return (observed_agreement - chance_agreement) / (1. - chance_agreement)


def observed_agreement_frequency(annotations1, annotations2, nclasses):
    """Observed frequency of agreement by two annotators.

    If a category is never observed, the frequency for that category is set
    to 0.0 .

    Only count entries where both annotators responded toward observed
    frequency.

    Arguments
    ---------
    annotations1 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    annotations2 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    weights_func : function(m_i, m_j)
        Weights function that receives two matrices of indices
        i, j and returns the matrix of weights between them.
        Default is :func:`~pyanno.measures.distances.diagonal_distance`

    Return
    ------
    result : float
        Observed agreement frequency value
    """

    conf_mat = confusion_matrix(annotations1, annotations2, nclasses)
    conf_mat_sum = conf_mat.sum()
    if conf_mat_sum != 0:
        observed_agreement = conf_mat.diagonal() / conf_mat_sum
    else:
        observed_agreement = np.empty((nclasses,), dtype=float)
        observed_agreement.fill(np.nan)
    return observed_agreement


def chance_agreement_same_frequency(annotations1, annotations2, nclasses):
    """Expected frequency of agreement by random annotations.

    Assumes that the annotators draw random annotations with the same
    frequency as the combined observed annotations.

    Arguments
    ---------
    annotations1 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    annotations2 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    weights_func : function(m_i, m_j)
        Weights function that receives two matrices of indices
        i, j and returns the matrix of weights between them.
        Default is :func:`~pyanno.measures.distances.diagonal_distance`

    Return
    ------
    result : float
        Chance agreement value
    """

    count1 = labels_count(annotations1, nclasses)
    count2 = labels_count(annotations2, nclasses)

    count_total = count1 + count2
    total = count_total.sum()
    chance_agreement = (count_total / total) ** 2.

    return chance_agreement


def chance_agreement_different_frequency(annotations1, annotations2, nclasses):
    """Expected frequency of agreement by random annotations.

    Assumes that the annotators draw annotations at random with different but
    constant frequencies.

    Arguments
    ---------
    annotations1 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    annotations2 : ndarray, shape = (n_items, )
        Array of annotations for a single annotator. Missing values should be
        indicated by :attr:`pyanno.util.MISSING_VALUE`

    weights_func : function(m_i, m_j)
        Weights function that receives two matrices of indices
        i, j and returns the matrix of weights between them.
        Default is :func:`~pyanno.measures.distances.diagonal_distance`

    Return
    ------
    result : float
        Chance agreement value
    """

    freq1 = labels_frequency(annotations1, nclasses)
    freq2 = labels_frequency(annotations2, nclasses)

    chance_agreement = freq1 * freq2
    return chance_agreement


def compute_nclasses(*annotations):
    """Infer the number of label classes from the data."""
    max_ = np.amax(list(map(np.amax, annotations)))
    return max_ + 1


def all_invalid(*annotations):
    """Return True if all annotations are invalid."""
    for anno in annotations:
        if np.any(is_valid(anno)):
            return False
    return True
