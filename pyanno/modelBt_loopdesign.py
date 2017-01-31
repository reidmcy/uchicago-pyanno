# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Andrey Rzhetsky
# License: Modified BSD license (2-clause)

"""This module defines model B-with-theta, optimized for a loop design.

The implementation assumes that there are a total or 8 annotators. Each item is
annotated by a triplet of annotators, according to the loop design described
in Rzhetsky et al., 2009.

E.g., for 16 items the loop design looks like this (`A` indicates a label,
`*` indicates a missing value): ::

    A A A * * * * *
    A A A * * * * *
    * A A A * * * *
    * A A A * * * *
    * * A A A * * *
    * * A A A * * *
    * * * A A A * *
    * * * A A A * *
    * * * * A A A *
    * * * * A A A *
    * * * * * A A A
    * * * * * A A A
    A * * * * * A A
    A * * * * * A A
    A A * * * * * A
    A A * * * * * A

"""

import numpy as np
import scipy.optimize
import scipy.stats
from traits.api import Int, Array
from pyanno.abstract_model import AbstractModel
from pyanno.sampling import optimize_step_size, sample_distribution
from pyanno.util import (random_categorical, compute_counts,
                         SMALLEST_FLOAT, MISSING_VALUE, labels_frequency,
                         is_valid, ninf_to_num)

import logging
logger = logging.getLogger(__name__)


# map of `n` to list of all possible triplets of `n` elements
_triplet_combinations = {}
def _get_triplet_combinations(n):
    """Return array of all possible combinations of n elements in triplets.
    """
    if n not in _triplet_combinations:
        _triplet_combinations[n] = (
            np.array([i for i in np.ndindex(n,n,n)]) )
    return _triplet_combinations[n]


class ModelBtLoopDesign(AbstractModel):
    """Implementation of Model B-with-theta from (Rzhetsky et al., 2009).

    The model assumes the existence of "true" underlying labels for each item,
    which are drawn from a categorical distribution, gamma. Annotators report
    these labels with some noise.

    This model is closely related to :class:`~ModelB`, but, crucially,
    the noise distribution is described by a small number of parameters (one
    per annotator), which makes their estimation efficient and less sensitive
    to local optima.

    These are the model parameters:

    - gamma[k] is the probability of label k

    - theta[j] parametrizes the probability that annotator `j` reports label
      `k'` given ground truth, `k`. More specifically,
      `P( annotator j chooses k' | real label = k)` is
      `theta[j]` for k' = k, or `(1 - theta[j]) / sum(theta)` if `k' != k `.

    This implementation is optimized for the loop design introduced in
    (Rzhetsky et al., 2009), which assumes that each item is annotated by 3
    out of 8 annotators. For a more general implementation, see
    :class:`~ModelBt`

    See the documentation for a more detailed description of the model.

    **Reference**

    * Rzhetsky A., Shatkay, H., and Wilbur, W.J. (2009). "How to get the most
      from your curation effort", PLoS Computational Biology, 5(5).
    """

    nclasses = Int
    nannotators = Int(8)
    # number of annotators rating each item in the loop design
    nannotators_per_item = Int(3)
    gamma = Array(dtype=float, shape=(None,))
    theta = Array(dtype=float, shape=(None,))

    def __init__(self, nclasses, gamma, theta, **traits):
        """Create an instance of ModelB.

        Arguments
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
        self.gamma = gamma
        self.theta = theta

        super(ModelBtLoopDesign, self).__init__(**traits)


    ##### Model and data generation methods ###################################

    @staticmethod
    def create_initial_state(nclasses, gamma=None, theta=None):
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
            number of categories

        gamma : nparray
            An array of floats with size that holds the probability of each
            annotation value. Default is None

        theta : nparray
            An array of floats that the parameters of P( v_i | psi ) (one for
            each annotator)

        Returns
        -------
        model : :class:`~ModelBtLoopDesign`
            Instance of ModelBtLoopDesign
        """

        if gamma is None:
            gamma = ModelBtLoopDesign._random_gamma(nclasses)

        if theta is None:
            nannotators = 8
            theta = ModelBtLoopDesign._random_theta(nannotators)

        model = ModelBtLoopDesign(nclasses, gamma, theta)
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
        nannotators = self.nannotators
        nitems = labels.shape[0]
        nitems_per_loop = np.ceil(float(nitems) / nannotators)

        annotations = np.empty((nitems, nannotators), dtype=int)
        for j in range(nannotators):
            for i in range(nitems):
                distr = self._theta_to_categorical(theta[j], labels[i])
                annotations[i,j]  = random_categorical(distr, 1)

        # mask annotation value according to loop design
        for l in range(nannotators):
            label_idx = np.arange(l+self.nannotators_per_item, l+nannotators) % 8
            annotations[l*nitems_per_loop:(l+1)*nitems_per_loop,
                        label_idx] = MISSING_VALUE

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

        # wrap log likelihood function to give it to optimize.fmin
        _llhood_counts = self._log_likelihood_counts
        def _wrap_llhood(params, counts):
            self.gamma, self.theta = self._vector_to_params(params)
            # minimize *negative* likelihood
            return - _llhood_counts(counts)

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

        # wrap log likelihood function to give it to optimize.fmin
        _llhood_counts = self._log_likelihood_counts
        _log_prior = self._log_prior
        def _wrap_llhood(params, counts):
            self.gamma, self.theta = self._vector_to_params(params)
            # minimize *negative* posterior probability of parameters
            return - (_llhood_counts(counts) + _log_prior())

        self._parameter_estimation(_wrap_llhood, annotations,
                                   estimate_gamma=estimate_gamma)


    def _parameter_estimation(self, objective, annotations,
                              estimate_gamma=True):
        counts = compute_counts(annotations, self.nclasses)

        params_start = self._random_initial_parameters(annotations,
                                                       estimate_gamma)

        logger.info('Start parameters optimization...')

        # TODO: use gradient, constrained optimization
        params_best = scipy.optimize.fmin(objective,
                                          params_start,
                                          args=(counts,),
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
            gamma = ModelBtLoopDesign._random_gamma(self.nclasses)

        theta = ModelBtLoopDesign._random_theta(self.nannotators)
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

        counts = compute_counts(annotations, self.nclasses)
        return self._log_likelihood_counts(counts)


    def _log_likelihood_counts(self, counts):
        """Compute the log likelihood of annotations given the model.

        This method assumes the data is in counts format.
        """

        # TODO: check if it's possible to replace these constraints with bounded optimization
        # check boundary conditions
        if (min(min(self.gamma), min(self.theta)) < 0.
            or max(max(self.gamma), max(self.theta)) > 1.):
            #return np.inf
            return SMALLEST_FLOAT

        llhood = 0.
        # loop over the 8 combinations of annotators
        for i in range(8):
            # extract the theta parameters for this triplet
            triplet_indices = np.arange(i, i+3) % self.nannotators
            triplet_indices.sort()
            theta_triplet = self.theta[triplet_indices]

            # compute the likelihood for the triplet
            llhood += self._log_likelihood_triplet(counts[:,i],
                                                   theta_triplet)

        return llhood


    def _log_likelihood_triplet(self, counts_triplet, theta_triplet):
        """Compute the log likelihood of data for one triplet of annotators.

        Input:
        counts_triplet -- count data for one combination of annotators
        theta_triplet -- theta parameters of the current triplet
        """

        # log \prod_n P(v_{ijk}^{n} | params)
        # = \sum_n log P(v_{ijk}^{n} | params)
        # = \sum_v_{ijk}  count(v_{ijk}) log P( v_{ijk} | params )
        #
        # where n is n-th annotation of triplet {ijk}]

        # compute P( v_{ijk} | params )
        pf = self._pattern_frequencies(theta_triplet)
        log_pf = ninf_to_num(np.log(pf))
        l = (counts_triplet * log_pf).sum()

        return l


    def _pattern_frequencies(self, theta_triplet):
        """Compute vector of P(v_{ijk}|params) for each combination of v_{ijk}.
        """

        gamma = self.gamma
        nclasses = self.nclasses
        # list of all possible combinations of v_i, v_j, v_k elements
        v_ijk_combinations = _get_triplet_combinations(nclasses)

        # P( v_{ijk} | params ) = \sum_psi P( v_{ijk} | psi, params ) P( psi )

        pf = 0.
        not_theta = (1.-theta_triplet) / (nclasses-1.)
        p_v_ijk_given_psi = np.empty_like(v_ijk_combinations, dtype=float)
        for psi in range(nclasses):
            for j in range(3):
                p_v_ijk_given_psi[:,j] = np.where(v_ijk_combinations[:,j]==psi,
                                                  theta_triplet[j],
                                                  not_theta[j])
            pf += p_v_ijk_given_psi.prod(1) * gamma[psi]
        return pf


    def _log_prior(self):
        """Compute log probability of prior on the theta parameters."""
        log_prob = scipy.stats.beta._logpdf(self.theta, 2., 1.).sum()
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

        # optimize step size
        counts = compute_counts(annotations, self.nclasses)

        # wrap log likelihood function to give it to optimize_step_size and
        # sample_distribution
        _llhood_counts = self._log_likelihood_counts
        _log_prior = self._log_prior
        def _wrap_llhood(params, counts):
            self.theta = params
            return _llhood_counts(counts) + _log_prior()

        # TODO this save-reset is rather ugly, refactor: create copy of
        #      model and sample over it
        # save internal parameters to reset at the end of sampling
        save_params = (self.gamma, self.theta)
        try:
            # compute optimal step size for given target rejection rate
            params_start = self.theta.copy()
            params_upper = np.ones((self.nannotators,))
            params_lower = np.zeros((self.nannotators,))
            step = optimize_step_size(_wrap_llhood, params_start, counts,
                                params_lower, params_upper,
                                step_optimization_nsamples,
                                adjust_step_every,
                                target_rejection_rate,
                                rejection_rate_tolerance)

            # draw samples from posterior distribution over theta
            samples = sample_distribution(_wrap_llhood, params_start, counts,
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

        nitems = annotations.shape[0]
        gamma = self.gamma
        nclasses = self.nclasses

        # get indices of annotators active in each row
        valid_entries = is_valid(annotations).nonzero()
        annotator_indices = np.reshape(valid_entries[1],
            (nitems, self.nannotators_per_item))
        valid_annotations = annotations[valid_entries]
        valid_annotations = np.reshape(valid_annotations,
            (nitems, self.nannotators_per_item))

        # thetas of active annotators
        theta_equal = self.theta[annotator_indices]
        theta_not_equal = (1. - theta_equal) / (nclasses - 1.)

        # compute posterior over psi
        psi_distr = np.zeros((nitems, nclasses))
        for psi in range(nclasses):
            tmp = np.where(valid_annotations == psi,
                           theta_equal, theta_not_equal)
            psi_distr[:,psi] = gamma[psi] * tmp.prod(1)

        # normalize distribution
        psi_distr /= psi_distr.sum(1)[:,np.newaxis]
        return psi_distr


    ##### Verify input ########################################################

    def are_annotations_compatible(self, annotations):
        """Check if the annotations are compatible with the models' parameters.
        """

        if not super(ModelBtLoopDesign, self).are_annotations_compatible(
            annotations):
            return False

        masked_annotations = np.ma.masked_equal(annotations, MISSING_VALUE)

        # exactly 3 annotations per row
        nvalid = (~masked_annotations.mask).sum(1)
        if not np.all(nvalid == self.nannotators_per_item):
            return False

        return True
