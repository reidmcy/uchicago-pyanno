# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

import unittest
import numpy as np
import pyanno.util as pu
from pyanno.util import MISSING_VALUE as MV

class TestUtil(unittest.TestCase):

    def test_random_categorical(self):
        distr = np.array([0.0, 0.3, 0.6, 0.05, 0.05])
        nsamples = 10000
        samples = pu.random_categorical(distr, nsamples)
        freq = np.bincount(samples) / float(nsamples)
        np.testing.assert_almost_equal(freq, distr, 2)


    def test_labels_frequency(self):
        annotations = np.array(
            [
                [ 1,  2, -2, -2],
                [-2, -2,  3,  3],
                [-2,  1,  3,  1],
                [-2, -2, -2, -2]
            ]
        )
        nclasses = 5
        expected = np.array([0., 3., 1., 3., 0.]) / 7.
        result = pu.labels_frequency(annotations, nclasses, missing_val=-2)
        np.testing.assert_equal(result, expected)


    def test_labels_count(self):
        annotations = np.array(
            [
                [ 1,  2, -2, -2],
                [-2, -2,  3,  3],
                [-2,  1,  3,  1],
                [-2, -2, -2, -2]
            ]
        )
        nclasses = 5
        expected = np.array([0., 3., 1., 3., 0.])
        result = pu.labels_count(annotations, nclasses, missing_val=-2)
        np.testing.assert_equal(result, expected)


    def test_majority_vote(self):
        annotations = np.array(
            [
                [1, 2, 2, MV],
                [2, 2, 2, 2],
                [1, 1, 3, 3],
                [1, 3, 3, 2],
                [MV, 2, 3, 1],
                [MV, MV, MV, 3]
            ]
        )
        expected = np.array([2, 2, 1, 3, 1, 3])
        result = pu.majority_vote(annotations)
        np.testing.assert_equal(expected, result)


    def test_majority_vote_empty_item(self):
        # Bug: majority vote with row of invalid annotations fails
        annotations = np.array(
            [[1, 2, 3],
             [MV, MV, MV],
             [1, 2, 2]]
        )
        expected = [1, MV, 2]
        result = pu.majority_vote(annotations)
        np.testing.assert_equal(expected, result)


if __name__ == '__main__':
    unittest.main()
