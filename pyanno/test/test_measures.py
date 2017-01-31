# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)



import unittest
import numpy as np

import pyanno
import pyanno.measures.agreement as pma
import pyanno.measures.covariation as pmc
import pyanno.measures.helpers as pmh
import pyanno.measures.distances as pmd

from pyanno.util import MISSING_VALUE as MV, is_valid, PyannoValueError


class Bunch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestMeasures(unittest.TestCase):

    def setUp(self):
        """Define fixtures."""

        # ---- annotations for fully agreeing annotators
        nitems = 100
        nannotators = 5
        nclasses = 4

        # perfect agreement, 2 missing annotations per row
        annotations = np.empty((nitems, nannotators), dtype=int)

        for i in range(nitems):
            annotations[i,:] = np.random.randint(nclasses)
            perm = np.random.permutation(nclasses)
            annotations[i,perm[0:2]] = MV

        self.full_agreement = Bunch(nitems=nitems,
                                    nannotators=nannotators,
                                    nclasses=nclasses,
                                    annotations=annotations)

        # ---- test example from
        # http://en.wikipedia.org/wiki/Krippendorff%27s_Alpha#A_computational_example
        annotations = np.array(
            [
                [MV,  1, MV],
                [MV, MV, MV],
                [MV,  2,  2],
                [MV,  1,  1],
                [MV,  3,  3],
                [ 3,  3,  4],
                [ 4,  4,  4],
                [ 1,  3, MV],
                [ 2, MV,  2],
                [ 1, MV,  1],
                [ 1, MV,  1],
                [ 3, MV,  3],
                [ 3, MV,  3],
                [MV, MV, MV],
                [ 3, MV,  4]
            ]
        )
        annotations[annotations>=0] -= 1
        coincidence = np.array(
            [
                [6, 0, 1, 0],
                [0, 4, 0, 0],
                [1, 0, 7, 2],
                [0, 0, 2, 3]
            ]
        )
        nc = np.array([7, 4, 10, 5])
        self.krippendorff_example = Bunch(annotations = annotations,
                                          coincidence = coincidence,
                                          nc = nc,
                                          nclasses = 4,
                                          alpha_diagonal = 0.811,
                                          alpha_binary = 0.691)

        # ---- Cronbach's alpha Excell example
        annotations = np.array(
            [
                [0, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [0, 0, 0, 1, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 1],
                [1, 1, 1, 0, 1],
                [0, 0, 1, 1, 1]
            ]
        )
        self.cronbach_example = Bunch(annotations = annotations,
                                      nclasses = 2,
                                      alpha = 0.619658)

        # ---- Cohen's kappa example from
        # http://r.789695.n4.nabble.com/Cohen-s-Kappa-for-beginners-td2229658.html
        annotations = np.array(
            [
                [1, 2, 3, 1],
                [1, 3, 3, 1]
            ]
        ).T - 1
        self.cohen_example = Bunch(annotations = annotations,
                                   nclasses = 3,
                                   kappa = 0.6)

        # ---- Test for annotations with no valid entry
        annotations = np.empty((10, 8), dtype=int)
        annotations.fill(MV)
        self.invalid_test = Bunch(annotations = annotations)


    def test_confusion_matrix(self):
        anno1 = np.array([0, 0, 1, 1, 2, 3])
        anno2 = np.array([0, 1, 1, 1, 2, 2])
        expected = np.array(
            [
                [1, 1, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0]
            ])
        cm = pmh.confusion_matrix(anno1, anno2, 4)
        np.testing.assert_array_equal(cm, expected)


    def test_confusion_matrix_missing(self):
        """Test confusion matrix with missing data."""
        anno1 = np.array([0, 0, 1, 1, MV, 3])
        anno2 = np.array([0, MV, 1, 1, 2, 2])
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0]
            ])
        cm = pmh.confusion_matrix(anno1, anno2, 4)
        np.testing.assert_array_equal(cm, expected)


    def test_chance_agreement_same_frequency(self):
        distr = np.array([0.1, 0.5, 0.4])
        anno1 = pyanno.util.random_categorical(distr, nsamples=10000)
        anno2 = pyanno.util.random_categorical(distr, nsamples=10000)

        expected = distr ** 2.
        freqs = pmh.chance_agreement_same_frequency(anno1, anno2, len(distr))

        np.testing.assert_allclose(freqs, expected, atol=1e-2, rtol=0.)


    def test_chance_agreement_different_frequency(self):
        distr1 = np.array([0.1, 0.5, 0.4])
        distr2 = np.array([0.6, 0.2, 0.2])
        anno1 = pyanno.util.random_categorical(distr1, nsamples=10000)
        anno2 = pyanno.util.random_categorical(distr2, nsamples=10000)

        expected = distr1 * distr2
        freqs = pmh.chance_agreement_different_frequency(anno1, anno2,
                                                         len(distr1))

        np.testing.assert_allclose(freqs, expected, atol=1e-2, rtol=0.)


    def test_observed_agreement(self):
        anno1 = np.array([0, 0, 1, 1, MV, 3])
        anno2 = np.array([0, MV, 1, 1, MV, 2])
        nvalid = np.sum(is_valid(anno1) & is_valid(anno2))

        expected = np.array([1., 2., 0., 0.]) / nvalid
        freqs = pmh.observed_agreement_frequency(anno1, anno2, 4)

        np.testing.assert_array_equal(freqs, expected)


    def test_cohens_kappa(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        self.assertAlmostEqual(pma.cohens_kappa(fa.annotations[:,0],
                                                fa.annotations[:,1]),
                               1.0, 6)


    def test_cohens_weighted_kappa(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        self.assertAlmostEqual(pma.cohens_weighted_kappa(fa.annotations[:,0],
                                                         fa.annotations[:,1]),
                               1.0, 6)


    def test_cohens_weighted_kappa2(self):
        # cohen's weighted kappa is the same as cohen's kappa when
        # the weights are 0. on the diagonal and 1. elsewhere
        anno1 = np.array([0, 0, 1, 2, 1, MV, 3])
        anno2 = np.array([0, MV, 1, 0, 1, MV, 2])
        weighted_kappa = pma.cohens_weighted_kappa(anno1, anno2,
                                                   pmd.binary_distance)
        cohens_kappa = pma.cohens_kappa(anno1, anno2)
        self.assertAlmostEqual(weighted_kappa, cohens_kappa, 6)


    def test_cohens_kappa2(self):
        ce = self.cohen_example

        kappa = pma.cohens_kappa(ce.annotations[:,0],
                                ce.annotations[:,1])
        self.assertAlmostEqual(kappa, ce.kappa, 6)


    def test_scotts_pi(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        self.assertAlmostEqual(pma.scotts_pi(fa.annotations[:,0],
                                             fa.annotations[:,1]),
                               1.0, 6)


    def test_fleiss_kappa_nannotations(self):
        # same example as
        # http://en.wikipedia.org/wiki/Fleiss%27_kappa#Worked_example
        nannotations = np.array(
            [
                [0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7]
            ]
        )
        expected = 0.21
        kappa = pma._fleiss_kappa_nannotations(nannotations)
        self.assertAlmostEqual(kappa, expected, 2)


    def test_fleiss_kappa(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        self.assertAlmostEqual(pma.fleiss_kappa(fa.annotations, fa.nclasses),
                               1.0, 6)

        # unequal number of annotators per row
        fa.annotations[0,:] = MV
        self.assertRaises(PyannoValueError, pma.fleiss_kappa, fa.annotations)


    def test_krippendorffs_alpha(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        self.assertAlmostEqual(pma.krippendorffs_alpha(fa.annotations),
                               1.0, 6)


    def test_coincidence_matrix(self):
        # test example from
        # http://en.wikipedia.org/wiki/Krippendorff%27s_Alpha#A_computational_example
        kr = self.krippendorff_example
        coincidence = pmh.coincidence_matrix(kr.annotations, kr.nclasses)
        np.testing.assert_allclose(coincidence, kr.coincidence)


    def test_krippendorffs_alpha2(self):
        # test example from
        # http://en.wikipedia.org/wiki/Krippendorff%27s_Alpha#A_computational_example
        kr = self.krippendorff_example

        alpha = pma.krippendorffs_alpha(kr.annotations,
                                        metric_func=pmd.binary_distance)
        self.assertAlmostEqual(alpha, kr.alpha_binary, 3)

        alpha = pma.krippendorffs_alpha(kr.annotations,
                                        metric_func=pmd.diagonal_distance)
        self.assertAlmostEqual(alpha, kr.alpha_diagonal, 3)


    def test_pearsons_rho(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        self.assertAlmostEqual(pmc.pearsons_rho(fa.annotations[:,0],
                                                fa.annotations[:,1]),
                               1.0, 6)


    def test_spearmans_rho(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        self.assertAlmostEqual(pmc.spearmans_rho(fa.annotations[:,0],
                                                 fa.annotations[:,1]),
                               1.0, 6)


    def test_cronbachs_alpha(self):
        # test basic functionality with full agreement, missing annotations
        fa = self.full_agreement

        alpha = pmc.cronbachs_alpha(fa.annotations)
        alpha *= (fa.nitems - 1.) / fa.nitems
        self.assertAlmostEqual(alpha, 1.0, 6)


    def test_cronbachs_alpha2(self):
        # test basic functionality with full agreement, missing annotations
        ce = self.cronbach_example

        self.assertAlmostEqual(pmc.cronbachs_alpha(ce.annotations),
                               ce.alpha, 6)


    def test_matrix(self):
        ke = self.krippendorff_example
        mat = pyanno.measures.pairwise_matrix(pma.cohens_kappa,
                                              ke.annotations, nclasses=4)
        self.assertAlmostEqual(mat[1,1], 1., 6)
        kappa = pma.cohens_kappa(ke.annotations[:,2], ke.annotations[:,0])
        self.assertAlmostEqual(mat[2,0], kappa)


    def test_all_invalid(self):
        # behavior: all measures should return np.nan for a set of
        # annotations with no valid entry

        anno = self.invalid_test.annotations

        self.assertTrue(
            np.isnan(pma.scotts_pi(anno[:,0], anno[:,1], nclasses=4))
        )

        self.assertTrue(
            np.isnan(pma.cohens_kappa(anno[:,0], anno[:,1], nclasses=3))
        )

        self.assertTrue(
            np.isnan(pma.cohens_weighted_kappa(anno[:,0], anno[:,1],
                                               nclasses=5))
        )

        self.assertTrue(
            np.isnan(pma.fleiss_kappa(anno, nclasses=4))
        )

        self.assertTrue(
            np.isnan(pma.krippendorffs_alpha(anno, nclasses=7))
        )

        self.assertTrue(
            np.isnan(pmc.pearsons_rho(anno[:,0], anno[:,1], nclasses=4))
        )

        self.assertTrue(
            np.isnan(pmc.spearmans_rho(anno[:,0], anno[:,1], nclasses=4))
        )

        self.assertTrue(
            np.isnan(pmc.cronbachs_alpha(anno, nclasses=4))
        )


if __name__ == '__main__':
    unittest.main()
