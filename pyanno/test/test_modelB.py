import unittest
import numpy as np
from numpy import testing
from pyanno.modelB import ModelB

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
        for n in xrange(nsamples):
            model = ModelB.create_initial_state(nclasses, nannotators,
                                                alpha, beta)
            pis[n,:] = model.pi
            thetas[n,...] = model.theta
        assert_is_dirichlet(pis, beta)
        for j in xrange(nannotators):
            for k in xrange(nclasses):
                assert_is_dirichlet(thetas[:,j,k,:], alpha[k,:])


    def test_generate_samples(self):
        nclasses = 4
        nannotators = 6
        nitems = 8
        model = ModelB.create_initial_state(nclasses, nannotators)

        nsamples = 1000
        labels = np.empty((nsamples, nitems), dtype=int)
        for i in xrange(nsamples):
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

        nsamples = 3000
        labels = np.arange(nclasses)

        annotations = np.empty((nsamples, nitems, nannotators), dtype=int)
        for i in xrange(nsamples):
            annotations[i,:,:] = model.generate_annotations(labels)

        for j in xrange(nannotators):
            for i in xrange(nitems):
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
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        # create a new, empty model and infer back the parameters
        model = ModelB(nclasses, nannotators)
        before_llhood = (model.log_likelihood(annotations)
                         + model._log_prior(model.pi, model.theta))
        model.map(annotations, epsilon=1e-3, max_epochs=1000)
        after_llhood = (model.log_likelihood(annotations)
                        + model._log_prior(model.pi, model.theta))

        testing.assert_allclose(model.pi, true_model.pi, atol=1e-2, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-2, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_map_stability(self):
        # test complex model, check that it is stable (converge back to optimum)
        nclasses, nannotators, nitems = 4, 10, 10000
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        # create a new model with the true parameters, plus noise
        theta = true_model.theta + np.random.normal(loc=true_model.theta,
                                                    scale=0.01/nclasses)
        pi = true_model.pi + np.random.normal(loc=true_model.pi,
                                              scale=0.01/nclasses)
        model = ModelB(nclasses, nannotators, pi, theta)
        model.map(annotations, epsilon=1e-3, max_epochs=1000)

        testing.assert_allclose(model.pi, true_model.pi, atol=1e-1, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-1, rtol=0.)


    def test_mle_estimation(self):
        # test simple model, check that we get to global optimum

        nclasses, nannotators, nitems = 2, 3, 10000
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        # create a new, empty model and infer back the parameters
        model = ModelB(nclasses, nannotators)
        before_llhood = model.log_likelihood(annotations)
        model.mle(annotations, epsilon=1e-3, max_epochs=1000)
        after_llhood = model.log_likelihood(annotations)

        testing.assert_allclose(model.pi, true_model.pi, atol=1e-2, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-2, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_missing_annotations(self):
        # test simple model, check that we get to global optimum
        # TODO test likelihood is increasing

        nclasses, nannotators, nitems = 2, 3, 10000
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)
        # remove about 10% of the annotations
        for i in range(nitems*nannotators//10):
            i = np.random.randint(nitems)
            j = np.random.randint(nannotators)
            annotations[i,j] = -1

        # create a new, empty model and infer back the parameters
        model = ModelB(nclasses, nannotators)
        before_llhood = (model.log_likelihood(annotations)
                         + model._log_prior(model.pi, model.theta))
        model.map(annotations, epsilon=1e-3, max_epochs=1000)
        after_llhood = (model.log_likelihood(annotations)
                        + model._log_prior(model.pi, model.theta))

        testing.assert_allclose(model.pi, true_model.pi, atol=1e-2, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-2, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_log_likelihood(self):
        # check that log likelihood is maximal at true parameters
        nclasses, nannotators, nitems = 3, 5, 1500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelB.create_initial_state(nclasses, nannotators)
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        max_llhood = true_model.log_likelihood(annotations)
        # perturb pi
        for _ in xrange(20):
            theta = true_model.theta
            pi = np.random.normal(loc=true_model.pi, scale=0.1)
            pi = np.clip(pi, 0.001, 1.)
            pi /= pi.sum()
            model = ModelB(nclasses, nannotators, pi=pi, theta=theta,
                           alpha=true_model.alpha, beta=true_model.beta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

        # perturb theta
        for _ in xrange(20):
            pi = true_model.pi
            theta = np.random.normal(loc=true_model.theta, scale=0.1)
            theta = np.clip(theta, 0.001, 1.)
            for j in xrange(nannotators):
                for k in xrange(nclasses):
                    theta[j,k,:] /= theta[j,k,:].sum()
            model = ModelB(nclasses, nannotators, pi=pi, theta=theta,
                           alpha=true_model.alpha, beta=true_model.beta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)


if __name__ == '__main__':
    unittest.main()