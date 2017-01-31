# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Andrey Rzhetsky,
#          Bob Carpenter
# License: Modified BSD license (2-clause)

"""Utility functions."""

import numpy as np
from numpy import log
from numpy.core import getlimits
from scipy.special import gammaln
import time

import logging
logger = logging.getLogger(__name__)

#: In annotations arrays, this is the value used to indicate missing values
MISSING_VALUE = -1

#: Smallest possible floating point number, somtimes used instead of -np.inf
#: to make numberical calculation return a meaningful value
SMALLEST_FLOAT = getlimits.finfo(np.float).min


class PyannoValueError(ValueError):
    """ValueError subclass raised by pyAnno functions and methods.
    """
    pass


def random_categorical(distr, nsamples):
    """Return an array of samples from a categorical distribution.

    Arguments
    ---------
    distr : ndarray
        distr[i] is the probability of item i

    nsamples : int
        Number of samples to draw from the distribution

    Returns
    -------
    samples : ndarray, shape = (n_samples, )
        Samples from the distribution
    """
    assert np.allclose(distr.sum(), 1., atol=1e-8)
    cumulative = distr.cumsum()
    return cumulative.searchsorted(np.random.random(nsamples))


def ninf_to_num(x):
    """Substitute -inf with smallest floating point number."""
    is_neg_inf = np.isneginf(x)
    x[is_neg_inf] = SMALLEST_FLOAT

    return x


def dirichlet_llhood(theta, alpha):
    """Compute the log likelihood of theta under Dirichlet(alpha).

    Arguments
    ---------
    theta : ndarray
        Categorical probability distribution. theta[i] is the probability
        of item i. Elements of the array have to sum to 1.0 (not forced for
        efficiency reasons)

    alpha : ndarray
        Parameters of the Dirichlet distribution

    Returns
    -------
    log_likelihood : float
        Log lihelihood of theta given alpha
    """

    # substitute -inf with SMALLEST_FLOAT, so that 0*log(0) is 0 when necessary
    log_theta = ninf_to_num(log(theta))

    #log_theta = np.nan_to_num(log_theta)
    return (gammaln(alpha.sum())
            - (gammaln(alpha)).sum()
            + ((alpha - 1.) * log_theta).sum())


# TODO remove default condition when x[i] == 0.
def normalize(x, dtype=float):
    """Returns a normalized distribution (sums to 1.0).

    If x consists only of zero element, the returned array has elements
    1/n , where n is the length of x.
    """
    x = np.asarray(x, dtype=dtype)
    z = x.sum()
    if z <= 0:
        x = np.ones_like(x)
        z = float(len(x))
    return x / z


def create_band_matrix(shape, diagonal_elements):
    """Create a symmetrical band matrix from a list of elements.

    Arguments
    ---------
    shape : int
        Width of the matrix

    diagonal_elements : list or array
        List of elements in the first row. If the list is smaller than `shape`,
        the last element is used to fill the the remaining items.
    """

    diagonal_elements = np.asarray(diagonal_elements)
    def diag(i,j):
        x = np.absolute(i-j)
        x = np.minimum(diagonal_elements.shape[0]-1, x).astype(int)
        return diagonal_elements[x]
    return np.fromfunction(diag, (shape, shape))


# TODO clean up and simplify and rename
def compute_counts(annotations, nclasses):
    """Transform annotation data in counts format.

    At the moment, it is hard coded for 8 annotators, 3 annotators active at
    any time.

    Arguments
    ---------
    annotations : ndarray, shape = (n_items, 8)
        Annotations array

    nclasses : int
        Numer of label classes

    Returns
    -------
    data : ndarray, shape = (n_classes^3, 9)
        data[i,m] is the number of times the combination of annotators
        number m voted according to pattern i
    """
    index = np.array([[0, 1, 2],
        [1, 2, 3],
        [2, 3, 4],
        [3, 4, 5],
        [4, 5, 6],
        [5, 6, 7],
        [0, 6, 7],
        [0, 1, 7]], int)
    m = annotations.shape[0]
    n = annotations.shape[1]
    annotations = np.asarray(annotations, dtype=int)

    if n != 8:
        raise ValueError('Number of annotators must be 8')

    # compute counts of 3-annotator patterns for 8 triplets of annotators
    data = np.zeros((nclasses ** 3, 8), dtype=int)

    # transform each triple of annotations into a code in base `nclasses`
    for i in range(m):
        ind = np.where(annotations[i, :] >= 0)

        code = annotations[i, ind[0][0]] * (nclasses ** 2) +\
               annotations[i, ind[0][1]] * nclasses +\
               annotations[i, ind[0][2]]

        # o = index of possible combination of annotators in the loop design
        o = -100
        for j in range(8):
            k = 0
            for l in range(3):
                if index[j, l] == ind[0][l]:
                    k += 1
            if k == 3:
                o = j

        if o >= 0:
            data[code, o] += 1
        else:
            logger.debug(str(code) + " " + str(ind) + " = homeless code")

    return data


def labels_count(annotations, nclasses, missing_val=MISSING_VALUE):
    """Compute the total count of labels in observed annotations.

    Arguments
    ---------
    annotations : ndarray, shape = (n_items, n_annotators)
        annotations[i,j] is the annotation made by annotator j on item i

    nclasses : int
        Number of label classes in `annotations`

    missing_val : int
        Value used to indicate missing values in the annotations.
        Default is :attr:`MISSING_VALUE`

    Returns
    -------
    count : ndarray, shape = (n_classes, )
        count[k] is the number of elements of class k in `annotations`
    """
    valid = annotations!=missing_val
    nobservations = valid.sum()

    if nobservations == 0:
        # no valid observations
        raise PyannoValueError('No valid annotations')

    return np.bincount(annotations[valid], minlength=nclasses)


def labels_frequency(annotations, nclasses, missing_val=MISSING_VALUE):
    """Compute the total frequency of labels in observed annotations.


    Arguments
    ---------
    annotations : ndarray, shape = (n_items, n_annotators)
        annotations[i,j] is the annotation made by annotator j on item i

    nclasses : int
        Number of label classes in `annotations`

    missing_val : int
        Value used to indicate missing values in the annotations.
        Default is :attr:`MISSING_VALUE`

    Returns
    -------
    freq : ndarray, shape = (n_classes, )
        freq[k] is the frequency of elements of class k in `annotations`, i.e.
        their count over the number of total of observed (non-missing) elements
    """
    valid = annotations!=missing_val
    nobservations = valid.sum()

    if nobservations == 0:
        # no valid observations
        raise PyannoValueError('No valid annotations')

    return (np.bincount(annotations[valid], minlength=nclasses)
            / float(nobservations))


def is_valid(annotations):
    """Return True if annotation is valid.

    An annotation is valid if it is not equal to the missing value,
    MISSING_VALUE.
    """
    return annotations != MISSING_VALUE


def majority_vote(annotations):
    """Compute an estimate of the real class by majority vote.

    In case of ties, return the class with smallest number.
    If a row only contains invalid entries, return `MISSING_VALUE`.

    Arguments
    ---------
    annotations : ndarray, shape = (n_items, n_annotators)
        annotations[i,j] is the annotation made by annotator j on item i

    Return
    ------
    vote : ndarray, shape = (n_items, )
        vote[i] is the majority vote estimate for item i
    """

    nitems = annotations.shape[0]
    valid = is_valid(annotations)

    vote = np.empty((nitems,), dtype=int)

    for i in range(nitems):
        if not np.any(valid[i,:]):
            # no valid entries on this row
            vote[i] = MISSING_VALUE
        else:
            count = np.bincount(annotations[i,valid[i,:]])
            vote[i] = count.argmax()

    return vote


def string_wrap(st, mode):
    st = str(st)

    if mode == 1:
        st = "\033[1;29m" + st + "\033[0m"
    elif mode == 2:
        st = "\033[1;34m" + st + "\033[0m"
    elif mode == 3:
        st = "\033[1;44m" + st + "\033[0m"
    elif mode == 4:
        st = "\033[1;35m" + st + "\033[0m"
    elif mode == 5:
        st = "\033[1;33;44m" + st + "\033[0m"
    elif mode == 5:
        st = "\033[1;47;34m" + st + "\033[0m"
    else:
        st = st + ' '
    return st


class benchmark(object):
    """Simple context manager to simplify benchmarking.

    Usage: ::

        with benchmark('fast computation'):
            do_something()
    """

    def __init__(self,name):
        self.name = name
    def __enter__(self):
        print('---- start ----')
        self.start = time.time()
    def __exit__(self,ty,val,tb):
        end = time.time()
        print('---- stop ----')
        print(("%s : %0.3f seconds" % (self.name, end-self.start)))
        return False
