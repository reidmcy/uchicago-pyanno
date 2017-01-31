# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

import unittest
import numpy as np
from numpy import testing
from pyanno.modelBt_loopdesign import ModelBtLoopDesign
from pyanno.util import MISSING_VALUE as MV, is_valid, PyannoValueError


class TestModelBtLoopDesign(unittest.TestCase):

    def test_mle_estimation(self):
        # test simple model, check that we get to global optimum
        nclasses, nitems = 3, 500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelBtLoopDesign.create_initial_state(nclasses)
        annotations = true_model.generate_annotations(nitems)

        # create a new, empty model and infer back the parameters
        model = ModelBtLoopDesign.create_initial_state(nclasses)
        before_llhood = model.log_likelihood(annotations)
        model.mle(annotations)
        after_llhood = model.log_likelihood(annotations)

        testing.assert_allclose(model.gamma, true_model.gamma, atol=1e-1, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-1, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_map_estimation(self):
        # test simple model, check that we get to global optimum
        nclasses, nitems = 3, 500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelBtLoopDesign.create_initial_state(nclasses)
        annotations = true_model.generate_annotations(nitems)

        # create a new, empty model and infer back the parameters
        model = ModelBtLoopDesign.create_initial_state(nclasses)
        before_obj = model.log_likelihood(annotations) + model._log_prior()
        model.map(annotations)
        after_obj = model.log_likelihood(annotations) + model._log_prior()

        testing.assert_allclose(model.gamma, true_model.gamma, atol=1e-1, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-1, rtol=0.)
        self.assertGreater(after_obj, before_obj)


    def test_log_likelihood(self):
        # check that log likelihood is maximal at true parameters
        nclasses, nitems = 3, 1500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelBtLoopDesign.create_initial_state(nclasses)
        annotations = true_model.generate_annotations(nitems)

        max_llhood = true_model.log_likelihood(annotations)
        # perturb gamma
        for _ in range(20):
            theta = true_model.theta
            gamma = np.random.normal(loc=true_model.gamma, scale=0.1)
            gamma = np.clip(gamma, 0., 1.)
            gamma /= gamma.sum()
            model = ModelBtLoopDesign(nclasses, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

        # perturb theta
        for _ in range(20):
            gamma = true_model.gamma
            theta = np.random.normal(loc=true_model.theta, scale=0.1)
            theta = np.clip(theta, 0., 1.)
            model = ModelBtLoopDesign(nclasses, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)


    def test_sampling_theta(self):
        nclasses, nitems = 3, 500*8
        nsamples = 1000

        # create random model (this is our ground truth model)
        true_model = ModelBtLoopDesign.create_initial_state(
            nclasses,
            theta=np.array([0.01, 0.1, 0.9, 0.7, 0.7, 0.7, 0.7, 0.7])
        )
        # create random data
        annotations = true_model.generate_annotations(nitems)

        # create a new model
        model = ModelBtLoopDesign.create_initial_state(nclasses)
        # get optimal parameters (to make sure we're at the optimum)
        model.map(annotations)

        # modify parameters, to give false start to sampler
        real_theta = model.theta.copy()
        model.theta = model._random_theta(model.nannotators)
        # save current parameters
        gamma_before, theta_before = model.gamma.copy(), model.theta.copy()
        samples = model.sample_posterior_over_accuracy(
            annotations,
            nsamples,
            burn_in_samples=100,
            thin_samples=2
        )
        # test: the mean of the sampled parameters is the same as the MLE one
        # (up to 3 standard deviations of the estimate sample distribution)
        testing.assert_array_less(np.absolute(samples.mean(0)-real_theta),
                                  3.*samples.std(0))

        # check that original parameters are intact
        testing.assert_equal(model.gamma, gamma_before)
        testing.assert_equal(model.theta, theta_before)


    def test_inference(self):
        # perfect annotation, check that inferred label is correct
        nclasses, nitems = 3, 50*8

        # create random model (this is our ground truth model)
        gamma = np.ones((nclasses,)) / float(nclasses)
        theta = np.ones((8,)) * 0.999
        true_model = ModelBtLoopDesign(nclasses, gamma, theta)
        # create random data
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations_from_labels(labels)

        posterior = true_model.infer_labels(annotations)
        testing.assert_allclose(posterior.sum(1), 1., atol=1e-6, rtol=0.)
        inferred = posterior.argmax(1)

        testing.assert_equal(inferred, labels)
        self.assertTrue(np.all(posterior[np.arange(nitems),inferred] > 0.999))

        # at chance annotation, disagreeing annotators: get back prior
        gamma = ModelBtLoopDesign._random_gamma(nclasses)
        theta = np.ones((8,)) / float(nclasses)
        model = ModelBtLoopDesign(nclasses, gamma, theta)

        data = np.array([[MV, 0, 1, 2, MV, MV, MV, MV,]])
        testing.assert_almost_equal(np.squeeze(model.infer_labels(data)),
                                    model.gamma, 6)


    def test_generate_annotations(self):
        # test to check that annotations are masked correctly when the number
        # of items is not divisible by the number of annotators
        nclasses, nitems = 5, 8*30+3

        model = ModelBtLoopDesign.create_initial_state(nclasses)
        annotations = model.generate_annotations(nitems)

        valid = is_valid(annotations)
        # check that on every row there are exactly 3 annotations
        self.assertTrue(np.all(valid.sum(1) == 3))


    def test_annotations_compatibility(self):
        nclasses = 3
        model = ModelBtLoopDesign.create_initial_state(nclasses)

        # test method that checks annotations compatibility
        anno = np.array([[MV, MV, 0, 0, 1, MV, MV, MV]])
        self.assertTrue(model.are_annotations_compatible(anno))

        anno = np.array([[MV, MV, 0, 0, 1, MV, MV, MV, MV]])
        self.assertFalse(model.are_annotations_compatible(anno))

        anno = np.array([[MV, MV, 0, 0, 3, MV, MV, MV]])
        self.assertFalse(model.are_annotations_compatible(anno))

        anno = np.array([[MV, MV, 0, 0, 2, 1, MV, MV]])
        self.assertFalse(model.are_annotations_compatible(anno))

        anno = np.array([[0, 0, MV, -2, MV, MV, MV, MV]])
        self.assertFalse(model.are_annotations_compatible(anno))


    def test_raise_error_on_incompatible_annotation(self):
        nclasses = 3
        model = ModelBtLoopDesign.create_initial_state(nclasses)
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
