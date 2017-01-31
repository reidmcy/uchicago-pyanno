# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

import unittest
import numpy as np
from numpy import testing
from pyanno.models import ModelB
from pyanno.util import MISSING_VALUE as MV, PyannoValueError, labels_frequency


def assert_is_distributions(distr, axis=0):
    """Check that input array represents a collection of distributions.
    """
    integral = distr.sum(axis=axis)
    testing.assert_allclose(integral,
                            np.ones_like(integral), rtol=0., atol=1e-7)


def assert_is_dirichlet(samples, alpha):
    """Checks that samples 'samples' are drawn from Dirichlet(alpha)."""
    assert len(samples.shape) == 2
    alpha0 = alpha.sum(-1)
    expected_mean = alpha / alpha0
    expected_var = expected_mean * (1. - expected_mean) / (alpha0 + 1.)
    testing.assert_allclose(samples.mean(0), expected_mean, rtol=0.1)
    testing.assert_allclose(samples.var(0),
                            expected_var, rtol=0.2)


class TestModelB(unittest.TestCase):

    def test_random_model(self):
        nclasses = 4
        nannotators = 6
        nitems = 8

        # check size of parameters
        model = ModelB.create_initial_state(nclasses, nannotators)
        self.assertEqual(model.pi.shape, (nclasses,))
        assert_is_distributions(model.pi)
        self.assertEqual(model.theta.shape, (nannotators, nclasses, nclasses))
        assert_is_distributions(model.theta, axis=2)

        # check mean and variance of distribution
        beta = np.array([10., 2., 30., 5.])
        alpha = np.random.randint(1, 30, size=(nclasses, nclasses)).astype(float)
        # collect random parameters
        nsamples = 1000
        pis = np.zeros((nsamples, nclasses))
        thetas = np.zeros((nsamples, nannotators, nclasses, nclasses))
        for n in range(nsamples):
            model = ModelB.create_initial_state(nclasses, nannotators,
                                                alpha, beta)
            pis[n,:] = model.pi
            thetas[n,...] = model.theta
        assert_is_dirichlet(pis, beta)
        for j in range(nannotators):
            for k in range(nclasses):
                assert_is_dirichlet(thetas[:,j,k,:], alpha[k,:])


    def test_generate_samples(self):
        nclasses = 4
        nannotators = 6
        nitems = 8
        model = ModelB.create_initial_state(nclasses, nannotators)

        nsamples = 1000
        labels = np.empty((nsamples, nitems), dtype=int)
        for i in range(nsamples):
            labels[i] = model.generate_labels(nitems)

        # NOTE here we make use of the fact that the prior is the same for all
        # items
        freq = (np.bincount(labels.flat, minlength=nclasses)
                / float(np.prod(labels.shape)))
        testing.assert_almost_equal(freq, model.pi, 2)


    def test_generate_annotations(self):
        nclasses = 4
        nannotators = 6
        nitems = 4
        model = ModelB.create_initial_state(nclasses, nannotators)

        # test functionality of generate_annotations method
        anno = model.generate_annotations(100)
        self.assertEqual(anno.shape[0], 100)
        self.assertTrue(model.are_annotations_compatible(anno))

        # check that returned annotations match the prior
        nsamples = 3000
        labels = np.arange(nclasses)

        annotations = np.empty((nsamples, nitems, nannotators), dtype=int)
        for i in range(nsamples):
            annotations[i,:,:] = model.generate_annotations_from_labels(labels)

        for j in range(nannotators):
            for i in range(nitems):
                # NOTE here we use the fact the the prior is the same for all
                # annotators
                tmp = annotations[:,i,j]
                freq = np.bincount(tmp, minlength=nclasses) / float(nsamples)
                testing.assert_almost_equal(freq,
                                            model.theta[j,labels[i],:], 1)


    def test_map_estimation(self):
        # test simple model, check that we get to global optimum

        nclasses, nannotators, nitems = 2, 3, 10000
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        annotations = true_model.generate_annotations(nitems)

        # create a new, empty model and infer back the parameters
        model = ModelB.create_initial_state(nclasses, nannotators)
        before_llhood = (model.log_likelihood(annotations)
                         + model._log_prior(model.pi, model.theta))
        model.map(annotations, epsilon=1e-3, max_epochs=1000)
        after_llhood = (model.log_likelihood(annotations)
                        + model._log_prior(model.pi, model.theta))

        # errors in the estimation are due to the the high uncertainty over
        # real labels, due to the relatively high error probability under the
        # prior
        testing.assert_allclose(model.pi, true_model.pi, atol=0.05, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta,
                                atol=0.05, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_map_stability(self):
        # test complex model, check that it is stable (converge back to optimum)
        nclasses, nannotators, nitems = 4, 10, 10000
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        annotations = true_model.generate_annotations(nitems)

        # create a new model with the true parameters, plus noise
        theta = true_model.theta + np.random.normal(loc=true_model.theta,
                                                    scale=0.01/nclasses)
        pi = true_model.pi + np.random.normal(loc=true_model.pi,
                                              scale=0.01/nclasses)
        model = ModelB(nclasses, nannotators, pi, theta)
        model.map(annotations, epsilon=1e-3, max_epochs=1000)

        testing.assert_allclose(model.pi, true_model.pi, atol=0.05, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta,
                                atol=0.05, rtol=0.)


    def test_mle_estimation(self):
        # test simple model, check that we get to global optimum

        nclasses, nannotators, nitems = 2, 3, 10000
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        annotations = true_model.generate_annotations(nitems)

        # create a new, empty model and infer back the parameters
        model = ModelB.create_initial_state(nclasses, nannotators)
        before_llhood = model.log_likelihood(annotations)
        model.mle(annotations, epsilon=1e-3, max_epochs=1000)
        after_llhood = model.log_likelihood(annotations)

        # errors in the estimation are due to the the high uncertainty over
        # real labels, due to the relatively high error probability under the
        # prior
        testing.assert_allclose(model.pi, true_model.pi, atol=0.07, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=0.07,
                                rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_missing_annotations(self):
        # test simple model, check that we get to global optimum

        nclasses, nannotators, nitems = 2, 3, 10000
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        annotations = true_model.generate_annotations(nitems)
        # remove about 10% of the annotations
        for _ in range(nitems*nannotators//10):
            i = np.random.randint(nitems)
            j = np.random.randint(nannotators)
            annotations[i,j] = MV

        # create a new, empty model and infer back the parameters
        model = ModelB.create_initial_state(nclasses, nannotators)
        before_llhood = (model.log_likelihood(annotations)
                         + model._log_prior(model.pi, model.theta))
        model.map(annotations, epsilon=1e-3, max_epochs=1000)
        after_llhood = (model.log_likelihood(annotations)
                        + model._log_prior(model.pi, model.theta))

        testing.assert_allclose(model.pi, true_model.pi, atol=1e-1, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta,
                                atol=1e-1, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_log_likelihood(self):
        # check that log likelihood is maximal at true parameters
        nclasses, nannotators, nitems = 3, 5, 1500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        annotations = true_model.generate_annotations(nitems)

        max_llhood = true_model.log_likelihood(annotations)
        # perturb pi
        for _ in range(20):
            theta = true_model.theta
            pi = np.random.normal(loc=true_model.pi, scale=0.1)
            pi = np.clip(pi, 0.001, 1.)
            pi /= pi.sum()
            model = ModelB(nclasses, nannotators, pi=pi, theta=theta,
                           alpha=true_model.alpha, beta=true_model.beta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

        # perturb theta
        for _ in range(20):
            pi = true_model.pi
            theta = np.random.normal(loc=true_model.theta, scale=0.1)
            theta = np.clip(theta, 0.001, 1.)
            for j in range(nannotators):
                for k in range(nclasses):
                    theta[j,k,:] /= theta[j,k,:].sum()
            model = ModelB(nclasses, nannotators, pi=pi, theta=theta,
                           alpha=true_model.alpha, beta=true_model.beta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)


    def test_inference(self):
        # perfect annotation, check that inferred label is correct
        nclasses, nitems = 3, 50*8
        nannotators = 12

        # create random model (this is our ground truth model)
        alpha = np.eye(nclasses)
        beta = np.ones((nclasses,)) * 1e10
        true_model = ModelB.create_initial_state(nclasses, nannotators,
                                                 alpha, beta)

        # create random data
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations_from_labels(labels)

        posterior = true_model.infer_labels(annotations)
        testing.assert_allclose(posterior.sum(1), 1., atol=1e-6, rtol=0.)

        inferred = posterior.argmax(1)
        testing.assert_equal(inferred, labels)
        self.assertTrue(np.all(posterior[np.arange(nitems),inferred] > 0.999))

        # at chance annotation, disagreeing annotators: get back prior
        pi = np.random.dirichlet(np.random.random(nclasses)*3)
        theta = np.ones((nannotators, nclasses, nclasses)) / nclasses
        model = ModelB(nclasses, nannotators, pi=pi, theta=theta)

        data = np.array([[MV, 0, 1, 2, MV, MV, MV, MV, MV, MV, MV, MV]])
        posterior = model.infer_labels(data)
        testing.assert_almost_equal(np.squeeze(posterior),
                                    model.pi, 6)


    def test_sampling_theta(self):
        nclasses, nitems = 3, 8*500
        nannotators = 5
        nsamples = 100

        # create random model (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        # create random data
        annotations = true_model.generate_annotations(nitems)
        # remove about 1/3 of the annotations
        for _ in range(nitems*nannotators//3):
            i = np.random.randint(nitems)
            j = np.random.randint(nannotators)
            annotations[i,j] = MV

        # create a new model
        model = ModelB.create_initial_state(nclasses, nannotators)
        # get optimal parameters (to make sure we're at the optimum)
        model.mle(annotations)

        # modify parameters, to give false start to sampler
        real_theta = model.theta.copy()
        model.theta = model._random_theta(nclasses, nannotators, model.alpha)
        # save current parameters
        pi_before, theta_before = model.pi.copy(), model.theta.copy()
        theta, pi, label = model.sample_posterior_over_accuracy(
            annotations,
            nsamples,
            burn_in_samples = 100,
            thin_samples = 2,
            return_all_samples = True
        )

        self.assertEqual(theta.shape[0], nsamples)
        self.assertEqual(pi.shape, (nsamples, nclasses))
        self.assertEqual(label.shape, (nsamples, nitems))

        # test: the mean of the sampled parameters is the same as the MLE one
        # (up to 3 standard deviations of the estimate sample distribution)
        testing.assert_array_less(np.absolute(theta.mean(0)-real_theta),
                                  3.*theta.std(0))

        # check that original parameters are intact
        testing.assert_equal(model.pi, pi_before)
        testing.assert_equal(model.theta, theta_before)


    def test_fix_map_nans(self):
        # bug is: when the number of classes in the annotations is smaller
        # than the one assumed by the model, the objective function of the
        # MAP estimation returns 'nan'

        true_nclasses = 3
        nannotators = 5
        true_model = ModelB.create_initial_state(true_nclasses, nannotators)
        annotations = true_model.generate_annotations(100)

        nclasses = 4
        model = ModelB.create_initial_state(nclasses, nannotators)
        # manually run a few EM iteration
        init_accuracy = 0.6
        mle_em_generator = model._map_em_step(annotations, init_accuracy)
        for i in range(3):
            objective, _, _, _ = next(mle_em_generator)

        self.assertFalse(np.isnan(objective))


    def test_annotations_compatibility(self):
        nclasses = 3
        nannotators = 5
        model = ModelB.create_initial_state(nclasses, nannotators)

        # test method that checks annotations compatibility
        anno = np.array([[0, 1, MV, MV, MV]])
        self.assertTrue(model.are_annotations_compatible(anno))

        anno = np.array([[0, 0, 0, 0]])
        self.assertFalse(model.are_annotations_compatible(anno))

        anno = np.array([[4, 0, 0, 0, 0]])
        self.assertFalse(model.are_annotations_compatible(anno))

        anno = np.array([[-2, MV, MV, MV, MV]])
        self.assertFalse(model.are_annotations_compatible(anno))


    def test_raise_error_on_incompatible_annotation(self):
        nclasses = 3
        nannotators = 8
        model = ModelB.create_initial_state(nclasses, nannotators)
        anno = np.array([[MV, MV, 0, 0, 7, MV, MV, MV]])

        with self.assertRaises(PyannoValueError):
            model.mle(anno)

        with self.assertRaises(PyannoValueError):
            model.map(anno)

        with self.assertRaises(PyannoValueError):
            model.sample_posterior_over_accuracy(anno, 10)

        with self.assertRaises(PyannoValueError):
            model.infer_labels(anno)

        with self.assertRaises(PyannoValueError):
            model.log_likelihood(anno)


if __name__ == '__main__':
    unittest.main()
