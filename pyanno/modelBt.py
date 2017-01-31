# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Andrey Rzhetsky
# License: Modified BSD license (2-clause)

"""This module defines model B-with-theta.

pyAnno includes another implementation of B-with-theta,
:py:mod:`pyanno.modelBt_loopdesign`, which is optimized for a loop design
where each item is annotated by 3 out of 8 annotators.
"""

import numpy as np
import scipy.optimize
import scipy.stats
from traits.api import Int, Array
from pyanno.abstract_model import AbstractModel
from pyanno.sampling import optimize_step_size, sample_distribution
from pyanno.util import (random_categorical,
                         SMALLEST_FLOAT, labels_frequency,
                         is_valid )

import logging
logger = logging.getLogger(__name__)


class ModelBt(AbstractModel):
    """Implementation of Model B-with-theta from (Rzhetsky et al., 2009).

    The model assumes the existence of "true" underlying labels for each item,
    which are drawn from a categorical distribution, gamma. Annotators report
    these labels with some noise, according to their accuracy, theta.

    This model is closely related to :class:`~ModelB`, but, crucially,
    the noise distribution is described by a small number of parameters (one
    per annotator), which makes their estimation efficient and less sensitive
    to local optima.

    These are the model parameters:

    - `gamma[k]` is the probability of label k

    - `theta[j]` parametrizes the probability that annotator `j` reports label
      `k'` given ground truth, `k`. More specifically,
      `P( annotator j chooses k' | real label = k)` is
      `theta[j]` for k' = k, or `(1 - theta[j]) / sum(theta)` if `k' != k `.

    See the documentation for a more detailed description of the model.

    For a version of this model optimized for the loop design described
    in (Rzhetsky et al., 2009), see :class:`~ModelBtLoopDesign`.

    **Reference**

    * Rzhetsky A., Shatkay, H., and Wilbur, W.J. (2009). "How to get the most
      from your curation effort", PLoS Computational Biology, 5(5).
    """


    ######## Model traits

    # number of label classes
    nclasses = Int

    # number of annotators
    nannotators = Int

    # gamma[k] is prior probability of class k
    gamma = Array(dtype=float, shape=(None,))

    # theta[j] parametrizes P(annotator j chooses : | real label = k)
    theta = Array(dtype=float, shape=(None,))


    def __init__(self, nclasses, nannotators,
                 gamma, theta, **traits):
        """Create an instance of ModelB.

        Parameters
        ----------
        nclasses : int
            Number of possible annotation classes

        nannotators : int
            Number of annotators

        gamma : ndarray, shape = (n_classes, )
            gamma[k] is the prior probability of label class k

        theta : ndarray, shape = (n_annotators, )
            theta[j] parametrizes the accuracy of annotator j. Specifically,
            `P( annotator j chooses k' | real label = k)` is
            `theta[j]` for k' = k, or `(1 - theta[j]) / sum(theta)`
            if `k' != k `.
        """

        self.nclasses = nclasses
        self.nannotators = nannotators

        self.gamma = gamma
        self.theta = theta

        super(ModelBt, self).__init__(**traits)


    ##### Model and data generation methods ###################################

    @staticmethod
    def create_initial_state(nclasses, nannotators, gamma=None, theta=None):
        """Factory method returning a model with random initial parameters.

        It is often more convenient to use this factory method over the
        constructor, as one does not need to specify the initial model
        parameters.

        The parameters theta and gamma, controlling accuracy and prevalence,
        are initialized at random as follows:

        :math:`\\theta_j \sim \mathrm{Uniform}(0.6, 0.95)`

        :math:`\gamma \sim \mathrm{Dirichlet}(2.0)`

        Arguments
        ---------
        nclasses : int
            Number of label classes

        nannotators : int
            Number of annotators

        gamma : ndarray, shape = (n_classes, )
            gamma[k] is the prior probability of label class k

        theta : ndarray, shape = (n_annotators, )
            theta[j] parametrizes the accuracy of annotator j. Specifically,
            `P( annotator j chooses k' | real label = k)` is
            `theta[j]` for k' = k, or `(1 - theta[j]) / sum(theta)`
            if `k' != k `.

        Returns
        -------
        model : :class:`~ModelBt`
            Instance of ModelBt
        """

        if gamma is None:
            gamma = ModelBt._random_gamma(nclasses)

        if theta is None:
            theta = ModelBt._random_theta(nannotators)

        model = ModelBt(nclasses, nannotators, gamma, theta)
        return model


    @staticmethod
    def _random_gamma(nclasses):
        beta = 2.*np.ones((nclasses,))
        return np.random.dirichlet(beta)


    @staticmethod
    def _random_theta(nannotators):
        return np.random.uniform(low=0.6, high=0.95,
                                 size=(nannotators,))


    def generate_labels(self, nitems):
        """Generate random labels from the model."""
        return random_categorical(self.gamma, nitems)


    def generate_annotations_from_labels(self, labels):
        """Generate random annotations from the model, given labels

        The method samples random annotations from the conditional probability
        distribution of annotations, :math:`x_i^j`
        given labels, :math:`y_i`.

        Arguments
        ----------
        labels : ndarray, shape = (n_items,), dtype = int
            Set of "true" labels

        Returns
        -------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i
        """
        theta = self.theta
        nitems = labels.shape[0]

        annotations = np.empty((nitems, self.nannotators), dtype=int)
        for j in range(self.nannotators):
            for i in range(nitems):
                distr = self._theta_to_categorical(theta[j], labels[i])
                annotations[i,j]  = random_categorical(distr, 1)

        return annotations


    def generate_annotations(self, nitems):
        """Generate a random annotation set from the model.

        Sample a random set of annotations from the probability distribution
        defined the current model parameters:

            1) Label classes are generated from the prior distribution, pi

            2) Annotations are generated from the conditional distribution of
               annotations given classes, parametrized by theta

        Arguments
        ---------
        nitems : int
            Number of items to sample

        Returns
        -------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i
        """
        labels = self.generate_labels(nitems)
        return self.generate_annotations_from_labels(labels)


    def _theta_to_categorical(self, theta, psi):
        """Returns P( v_i = psi | theta_i ) as a distribution."""
        distr = np.empty((self.nclasses,))
        distr.fill((1.-theta)/(self.nclasses-1.))
        distr[psi] = theta
        assert np.allclose(distr.sum(), 1.)
        return distr


    ##### Parameters estimation methods #######################################

    def mle(self, annotations, estimate_gamma=True):
        """Computes maximum likelihood estimate (MLE) of parameters.

        Estimate the parameters :attr:`theta` and :attr:`gamma` from a set of
        observed annotations using maximum likelihood estimation.

        Arguments
        ----------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        estimate_gamma : bool
            If True, the parameters :attr:`gamma` are estimated by the empirical
            class frequency. If False, :attr:`gamma` is left unchanged.
        """
        self._raise_if_incompatible(annotations)

        # mask missing annotations
        missing_mask_nclasses = self._missing_mask(annotations)

        # wrap log likelihood function to give it to optimize.fmin
        _llhood = self._log_likelihood_core
        def _wrap_llhood(params):
            gamma, theta = self._vector_to_params(params)
            # minimize *negative* likelihood
            return - _llhood(annotations, gamma, theta, missing_mask_nclasses)

        self._parameter_estimation(_wrap_llhood, annotations,
                                   estimate_gamma=estimate_gamma)


    def map(self, annotations, estimate_gamma=True):
        """Computes maximum a posteriori (MAP) estimate of parameters.

        Estimate the parameters :attr:`theta` and :attr:`gamma` from a set of
        observed annotations using maximum a posteriori estimation.

        Arguments
        ----------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        estimate_gamma : bool
            If True, the parameters :attr:`gamma` are estimated by the empirical
            class frequency. If False, :attr:`gamma` is left unchanged.
        """
        self._raise_if_incompatible(annotations)

        # mask missing annotations
        missing_mask_nclasses = self._missing_mask(annotations)

        # wrap objective function to give it to optimize.fmin
        _llhood = self._log_likelihood_core
        _log_prior = self._log_prior
        def _wrap_llhood(params):
            gamma, theta = self._vector_to_params(params)
            # minimize *negative* posterior probability of parameters
            return - (_llhood(annotations, gamma, theta, missing_mask_nclasses)
                      + _log_prior(theta))

        self._parameter_estimation(_wrap_llhood, annotations,
                                   estimate_gamma=estimate_gamma)


    def _parameter_estimation(self, objective, annotations,
                              estimate_gamma=True):
        params_start = self._random_initial_parameters(annotations,
                                                       estimate_gamma)

        logger.info('Start parameters optimization...')

        # TODO: use gradient, constrained optimization
        params_best = scipy.optimize.fmin(objective,
                                          params_start,
                                          xtol=1e-4, ftol=1e-4,
                                          disp=False, maxiter=10000)

        logger.info('Parameters optimization finished')

        # parse arguments and update
        self.gamma, self.theta = self._vector_to_params(params_best)


    def _random_initial_parameters(self, annotations, estimate_gamma):
        if estimate_gamma:
            # estimate gamma from observed annotations
            gamma = labels_frequency(annotations, self.nclasses)
        else:
            gamma = ModelBt._random_gamma(self.nclasses)

        theta = ModelBt._random_theta(self.nannotators)
        return self._params_to_vector(gamma, theta)


    def _params_to_vector(self, gamma, theta):
        """Convert the tuple (gamma, theta) to a parameters vector.

        Used to interface with the optimization routines.
        """
        return np.r_[gamma[:-1], theta]


    def _vector_to_params(self, params):
        """Convert a parameters vector to (gamma, theta) tuple.

        Used to interface with the optimization routines.
        """
        nclasses = self.nclasses
        gamma = np.zeros((nclasses,))
        gamma[:nclasses-1] = params[:nclasses-1]
        gamma[-1] = 1. - gamma[:nclasses-1].sum()
        theta = params[nclasses-1:]
        return gamma, theta


    ##### Model likelihood methods ############################################

    def log_likelihood(self, annotations):
        """Compute the log likelihood of a set of annotations given the model.

        Returns :math:`\log P(\mathbf{x} | \gamma, \\theta)`,
        where :math:`\mathbf{x}` is the array of annotations.

        Arguments
        ----------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        Returns
        -------
        log_lhood : float
            log likelihood of `annotations`
        """

        self._raise_if_incompatible(annotations)

        # mask missing annotations
        missing_mask_nclasses = self._missing_mask(annotations)

        return self._log_likelihood_core(annotations,
                                         self.gamma, self.theta,
                                         missing_mask_nclasses)


    def _log_likelihood_core(self, annotations,
                             gamma, theta,
                             missing_mask_nclasses):

        # check boundary conditions
        if (min(min(gamma), min(theta)) < 0.
            or max(max(gamma), max(theta)) > 1.):
            #return -np.inf
            return SMALLEST_FLOAT

        unnorm_category = self._compute_category(
            annotations, gamma, theta,
            missing_mask_nclasses=missing_mask_nclasses,
            normalize=False
        )

        llhood = np.log(unnorm_category.sum(1)).sum()
        if np.isnan(llhood):
            llhood = SMALLEST_FLOAT

        return llhood


    def _compute_category(self, annotations, gamma, theta,
                          missing_mask_nclasses=None, normalize=True):
        """Compute P(category[i] = k | model, annotations).

        Arguments
        ---------
        annotations : ndarray
            Array of annotations

        gamma : ndarray
            Gamma parameters

        theta : ndarray
            Theta parameters

        missing_mask_nclasses : ndarray, shape=(nitems, nannotators, n_classes)
            Mask with True at missing values, tiled in the third dimension.
            If None, it is computed, but it can be specified to speed-up
            computations.

        normalize : bool
            If False, do not normalize the distribution.

        Returns
        -------
        category : ndarray, shape = (n_items, n_classes)
            category[i,k] is the (unnormalized) probability of class k for
            item i
        """

        nitems, nannotators = annotations.shape

        # compute mask of invalid entries in annotations if necessary
        if missing_mask_nclasses is None:
            missing_mask_nclasses = self._missing_mask(annotations)

        accuracy = self._accuracy_tensor(theta)

        # unnorm_category is P(category[i] = k | model, data), unnormalized
        unnorm_category = np.tile(gamma.copy(), (nitems, 1))
        # mask missing annotations
        annotators = np.arange(nannotators)[None, :]
        tmp = np.ma.masked_array(accuracy[annotators, :, annotations],
                                 mask=missing_mask_nclasses)
        unnorm_category *= tmp.prod(1)

        if normalize:
            return unnorm_category / unnorm_category.sum(1)[:,None]

        return unnorm_category


    def _accuracy_tensor(self, theta):
        """Return the accuracy tensor.

        theta[j,k,k'] = P( annotator j emits k' | real class is k)
        """
        nannotators = self.nannotators
        nclasses = self.nclasses

        accuracy = np.empty((nannotators, nclasses, nclasses))
        for j in range(nannotators):
            for k in range(nclasses):
                accuracy[j,k,:] = self._theta_to_categorical(theta[j], k)

        return accuracy


    def _missing_mask(self, annotations):
        missing_mask = ~ is_valid(annotations)
        missing_mask_nclasses = np.tile(missing_mask[:, :, None],
            (1, 1, self.nclasses))
        return missing_mask_nclasses


    def _log_prior(self, theta=None):
        """Compute log probability of prior on the theta parameters."""
        if theta is None:
            theta = self.theta
        log_prob = scipy.stats.beta._logpdf(theta, 2., 1.).sum()
        return log_prob


    ##### Sampling posterior over parameters ##################################

    def sample_posterior_over_accuracy(self, annotations, nsamples,
                                       burn_in_samples = 100,
                                       thin_samples = 5,
                                       target_rejection_rate = 0.3,
                                       rejection_rate_tolerance = 0.2,
                                       step_optimization_nsamples = 500,
                                       adjust_step_every = 100):
        """Return samples from posterior distribution over theta given data.

        Samples are drawn using a variant of a Metropolis-Hasting Markov Chain
        Monte Carlo (MCMC) algorithm. Sampling proceeds in two phases:

            1) *step size estimation phase*: first, the step size in the
               MCMC algorithm is adjusted to achieve a given rejection rate.

            2) *sampling phase*: second, samples are collected using the
               step size from phase 1.

        Arguments
        ----------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        nsamples : int
            Number of samples to return (i.e., burn-in and thinning samples
            are not included)

        burn_in_samples : int
            Discard the first `burn_in_samples` during the initial burn-in
            phase, where the Monte Carlo chain converges to the posterior

        thin_samples : int
            Only return one every `thin_samples` samples in order to reduce
            the auto-correlation in the sampling chain. This is called
            "thinning" in MCMC parlance.

        target_rejection_rate : float
            target rejection rate for the step size estimation phase

        rejection_rate_tolerance : float
            the step size estimation phase is ended when the rejection rate for
            all parameters is within `rejection_rate_tolerance` from
            `target_rejection_rate`

        step_optimization_nsamples : int
            number of samples to draw in the step size estimation phase

        adjust_step_every : int
            number of samples after which the step size is adjusted during
            the step size estimation pahse

        Returns
        -------
        samples : ndarray, shape = (n_samples, n_annotators)
            samples[i,:] is one sample from the posterior distribution over the
            parameters `theta`
        """

        self._raise_if_incompatible(annotations)
        nsamples = self._compute_total_nsamples(nsamples,
                                                burn_in_samples,
                                                thin_samples)

        # mask missing annotations
        missing_mask_nclasses = self._missing_mask(annotations)

        # wrap objective function to give it to optimize_step_size and
        # sample_distribution
        _llhood = self._log_likelihood_core
        _log_prior = self._log_prior
        gamma = self.gamma
        def _wrap_llhood(params, args):
            theta = params
            return (_llhood(annotations, gamma, theta, missing_mask_nclasses)
                    + _log_prior(theta))

        # TODO this save-reset is rather ugly, refactor: create copy of
        #      model and sample over it
        # save internal parameters to reset at the end of sampling
        save_params = (self.gamma, self.theta)
        try:
            # compute optimal step size for given target rejection rate
            params_start = self.theta.copy()
            params_upper = np.ones((self.nannotators,))
            params_lower = np.zeros((self.nannotators,))
            step = optimize_step_size(_wrap_llhood, params_start, None,
                                params_lower, params_upper,
                                step_optimization_nsamples,
                                adjust_step_every,
                                target_rejection_rate,
                                rejection_rate_tolerance)

            # draw samples from posterior distribution over theta
            samples = sample_distribution(_wrap_llhood, params_start, None,
                                          step, nsamples,
                                          params_lower, params_upper)
            return self._post_process_samples(samples, burn_in_samples,
                                              thin_samples)
        finally:
            # reset parameters
            self.gamma, self.theta = save_params


    ##### Posterior distributions #############################################

    def infer_labels(self, annotations):
        """Infer posterior distribution over label classes.

         Compute the posterior distribution over label classes given observed
         annotations, :math:`P( \mathbf{y} | \mathbf{x}, \\theta, \omega)`.

         Arguments
         ----------
         annotations : ndarray, shape = (n_items, n_annotators)
             annotations[i,j] is the annotation of annotator j for item i

         Returns
         -------
         posterior : ndarray, shape = (n_items, n_classes)
             posterior[i,k] is the posterior probability of class k given the
             annotation observed in item i.
         """

        self._raise_if_incompatible(annotations)

        category = self._compute_category(annotations,
                                          self.gamma,
                                          self.theta)

        return category
