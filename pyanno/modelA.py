# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Andrey Rzhetsky
# License: Modified BSD license (2-clause)

"""This module defines the class ModelA, an implementation of model A from
Rzhetsky et al., 2009.

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

**Reference**

* Rzhetsky A., Shatkay, H., and Wilbur, W.J. (2009). "How to get the most from
  your curation effort", PLoS Computational Biology, 5(5).
"""

from collections import defaultdict
import numpy as np
import scipy.optimize
import scipy.stats
from traits.trait_numeric import Array
from traits.trait_types import Int
from pyanno.abstract_model import AbstractModel
from pyanno.sampling import optimize_step_size, sample_distribution
from pyanno.util import (compute_counts, random_categorical,
                         labels_frequency, MISSING_VALUE, SMALLEST_FLOAT,
                         ninf_to_num )

import logging
logger = logging.getLogger(__name__)


_compatibility_tables_cache = {}
def _compatibility_tables(nclasses):
    """Return a map from agreement indices to annotation patterns.

    The agreement indices are defined as in Table 3 of
    Rzhetsky et al., 2009, supplementary material:
    0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@

    The dictionary maps an agreement index to an array of annotations
    compatible with the corresponding agreement pattern.
    For example, for nclasses=3 and index=1 the array contains these
    annotations:
    0 0 1
    0 0 2
    1 1 0
    1 1 2
    2 2 0
    2 2 1
    """

    if nclasses not in _compatibility_tables_cache:
        compatibility = defaultdict(list)

        # aaa
        for psi1 in range(nclasses):
            compatibility[0].append([psi1, psi1, psi1])

        # aaA, aAa, Aaa
        for psi1 in range(nclasses):
            for psi2 in range(nclasses):
                if psi2 == psi1: continue
                compatibility[1].append([psi1, psi1, psi2])
                compatibility[2].append([psi1, psi2, psi1])
                compatibility[3].append([psi2, psi1, psi1])

        # Aa@
        for psi1 in range(nclasses):
            for psi2 in range(nclasses):
                for psi3 in range(nclasses):
                    if psi1==psi2 or psi2==psi3 or psi1==psi3: continue
                    compatibility[4].append([psi1, psi2, psi3])

        for idx in range(5):
            compatibility[idx] = np.array(compatibility[idx], dtype=int)

        _compatibility_tables_cache[nclasses] = compatibility

    return _compatibility_tables_cache[nclasses]


def _triplet_to_counts_index(triplet, nclasses):
    """Map annotator triplets to data indices for the counts format.
    """
    return (triplet * np.array([nclasses**2, nclasses, 1])).sum(1)


class ModelA(AbstractModel):
    """Implementation of Model A from (Rzhetsky et al., 2009).

    The model defines a probability distribution over data annotations
    in which each item is annotated by three users. The distributions is
    described according to a three-steps generative model:

        1. First, the model independently generates correctness values for the
        triplet of annotators (e.g., CCI where C=correct, I=incorrect)

        2. Second, the model generates an agreement pattern compatible with
        the correctness values (e.g., CII is compatible with the agreement
        patterns 'abb' and 'abc', where different letters correspond to
        different annotations

        3. Finally, the model generates actual observations compatible with
        the agreement patterns

    The model has two main sets of parameters:

        - theta[j] is the probability that annotator j is correct

        - omega[k] is the probability of observing an annotation of class `k`
          over all items and annotators

    At the moment the implementation of the model assumes 1) a total of 8
    annotators, and 2) each item is annotated by exactly 3 annotators.

    See the documentation for a more detailed description of the model.

    **Reference**

    * Rzhetsky A., Shatkay, H., and Wilbur, W.J. (2009). "How to get the most
      from your curation effort", PLoS Computational Biology, 5(5).
    """


    ######## Model traits

    # number of label classes
    nclasses = Int

    # number of annotators
    nannotators = Int(8)

    # number of annotators rating each item in the loop design
    nannotators_per_item = Int(3)

    #### Model parameters

    # theta[j] is the probability that annotator j is correct
    theta = Array(dtype=float, shape=(None,))

    # omega[k] is the probability of observing label class k
    omega = Array(dtype=float, shape=(None,))


    def __init__(self, nclasses, theta, omega, **traits):
        """Create an instance of ModelA.

        Arguments
        ---------
        nclasses : int
            Number of possible annotation classes

        theta : ndarray, shape = (n_annotators, )
            theta[j] is the probability of annotator j being correct

        omega : ndarray, shape = (n_classes, )
            omega[k] is the probability of observing a label of class k
        """

        self.nclasses = nclasses
        self.theta = theta
        self.omega = omega

        super(ModelA, self).__init__(**traits)


    ##### Model and data generation methods ###################################

    @staticmethod
    def create_initial_state(nclasses, theta=None, omega=None):
        """Factory method to create a new model.

        It is often more convenient to use this factory method over the
        constructor, as one does not need to specify the initial model
        parameters.

        If not specified, the parameters theta are drawn from a uniform
        distribution between 0.6 and 0.95 . The parameters omega are drawn
        from a Dirichlet distribution with parameters 2.0 :

        :math:`\\theta_j \sim \mathrm{Uniform}(0.6, 0.95)`

        :math:`\omega_k \sim \mathrm{Dirichlet}(2.0)`


        Arguments
        ---------
        nclasses : int
            number of possible annotation classes

        theta : ndarray, shape = (n_annotators, )
            theta[j] is the probability of annotator j being correct

        omega : ndarray, shape = (n_classes, )
            omega[k] is the probability of observing a label of class k
        """

        if theta is None:
            nannotators = 8
            theta = ModelA._random_theta(nannotators)

        if omega is None:
            omega = ModelA._random_omega(nclasses)

        return ModelA(nclasses, theta, omega)


    @staticmethod
    def _random_theta(nannotators):
        return np.random.uniform(low=0.6, high=0.95,
                                 size=(nannotators,))


    @staticmethod
    def _random_omega(nclasses):
        beta = 2.*np.ones((nclasses,))
        return np.random.dirichlet(beta)


    def generate_annotations(self, nitems):
        """Generate random annotations from the model.

        The method samples random annotations from the probability
        distribution defined by the model parameters:

            1) generate correct/incorrect labels for the three annotators,
               according to the parameters `theta`

            2) generate agreement patterns (which annotator agrees which whom)
               given the correctness information and the parameters `alpha`

            3) generate the annotations given the agreement patterns and the
               parameters `omega`


        Note that, according to the model's definition, only three annotators
        per item return an annotation. Non-observed annotations have the
        standard value of :attr:`~pyanno.util.MISSING_VALUE`.

        Arguments
        ---------
        nitems : int
            number of annotations to draw from the model

        Returns
        -------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i
        """
        theta = self.theta
        nannotators = self.nannotators
        nitems_per_loop = np.ceil(float(nitems) / nannotators)

        annotations = np.empty((nitems, nannotators), dtype=int)
        annotations.fill(MISSING_VALUE)

        # loop over annotator triplets (loop design)
        for j in range(nannotators):
            triplet_indices = np.arange(j, j+3) % self.nannotators
            start_idx = j*nitems_per_loop
            stop_idx = min(nitems, (j+1)*nitems_per_loop)
            nitems_this_loop = stop_idx - start_idx

            # -- step 1: generate correct / incorrect labels

            # parameters for this triplet
            theta_triplet = self.theta[triplet_indices]
            incorrect = self._generate_incorrectness(nitems_this_loop,
                                                     theta_triplet)

            # -- step 2: generate agreement patterns given correctness
            # convert boolean correctness into combination indices
            # (indices as in Table 3)
            agreement = self._generate_agreement(incorrect)

            # -- step 3: generate annotations
            annotations[start_idx:stop_idx,triplet_indices] = (
                self._generate_annotations(agreement)
                )

        return annotations


    def _generate_incorrectness(self, n, theta_triplet):
        _rnd = np.random.rand(n, self.nannotators_per_item)
        incorrect = _rnd >= theta_triplet
        return incorrect


    def _generate_agreement(self, incorrect):
        """Return indices of agreement pattern given correctness pattern.

        The indices returned correspond to agreement patterns
        as in Table 3: 0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@
        """

        # create tensor A_ijk
        # (cf. Table 3 in Rzhetsky et al., 2009, suppl. mat.)
        alpha = self._compute_alpha()
        agreement_tbl = np.array(
            [[1.,       0.,       0.,       0.,       0.],
             [0.,       1.,       0.,       0.,       0.],
             [0.,       0.,       1.,       0.,       0.],
             [0.,       0.,       0.,       1.,       0.],
             [0.,       0.,       0.,       alpha[0], 1.-alpha[0]],
             [0.,       0.,       alpha[1], 0.,       1.-alpha[1]],
             [0.,       alpha[2], 0.,       0.,       1.-alpha[2]],
             [alpha[3], alpha[4], alpha[5], alpha[6], 1.-alpha[3:].sum()]])

        # this array maps boolean correctness patterns (e.g., CCI) to
        # indices in the agreement tensor, `agreement_tbl`
        correctness_to_agreement_idx = np.array([0, 3, 2, 6, 1, 5, 4, 7])

        # convert correctness pattern to index in the A_ijk tensor
        correct_idx = correctness_to_agreement_idx[
                      incorrect[:,0]*1 + incorrect[:,1]*2 + incorrect[:,2]*4]

        # the indices stored in `agreement` correspond to agreement patterns
        # as in Table 3: 0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@
        nitems_per_loop = incorrect.shape[0]
        agreement = np.empty((nitems_per_loop,), dtype=int)
        for i in range(nitems_per_loop):
            # generate agreement pattern according to A_ijk
            agreement[i] = random_categorical(
                agreement_tbl[correct_idx[i]], 1)

        return agreement


    def _generate_annotations(self, agreement):
        """Generate triplet annotations given agreement pattern."""
        nitems_per_loop = agreement.shape[0]
        omega = self.omega
        annotations = np.empty((nitems_per_loop, 3), dtype=int)

        for i in range(nitems_per_loop):
            # get all compatible annotations
            compatible = _compatibility_tables(self.nclasses)[agreement[i]]
            # compute probability of each possible annotation
            distr = omega[compatible].prod(1)
            distr /= distr.sum()
            # draw annotation
            compatible_idx = random_categorical(distr, 1)[0]
            annotations[i,:] = compatible[compatible_idx, :]
        return annotations


    ##### Parameters estimation methods #######################################

    def mle(self, annotations, estimate_omega=True):
        """Computes maximum likelihood estimate (MLE) of parameters.

        Estimate the parameters :attr:`theta` and :attr:`omega` from a set of
        observed annotations using maximum likelihood estimation.

        Arguments
        ---------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        estimate_omega : bool
            If True, the parameters :attr:`omega` are estimated by the empirical
            class frequency. If False, :attr:`omega` is left unchanged.
        """

        self._raise_if_incompatible(annotations)

        def _wrap_lhood(params, counts):
            self.theta = params
            return - self._log_likelihood_counts(counts)

        self._parameter_estimation(_wrap_lhood, annotations,
                                   estimate_omega=estimate_omega)


    def map(self, annotations, estimate_omega=True):
        """Computes maximum a posteriori (MAP) estimate of parameters.

        Estimate the parameters :attr:`theta` and :attr:`omega` from a set of
        observed annotations using maximum a posteriori estimation.

        Arguments
        ---------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        estimate_omega : bool
            If True, the parameters :attr:`omega` are estimated by the empirical
            class frequency. If False, :attr:`omega` is left unchanged.
        """

        self._raise_if_incompatible(annotations)

        def _wrap_lhood(params, counts):
            self.theta = params
            return - (self._log_likelihood_counts(counts)
                      + self._log_prior())

        self._parameter_estimation(_wrap_lhood, annotations,
                                   estimate_omega=estimate_omega)


    def _parameter_estimation(self, objective, annotations,
                              estimate_omega=True):

        counts = compute_counts(annotations, self.nclasses)

        params_start, omega = self._random_initial_parameters(annotations,
                                                              estimate_omega)
        self.omega = omega

        logger.info('Start parameters optimization...')

        params_best = scipy.optimize.fmin(objective,
                                          params_start,
                                          args=(counts,),
                                          xtol=1e-4, ftol=1e-4,
                                          disp=False,
                                          maxiter=10000)

        logger.info('Parameters optimization finished')

        self.theta = params_best


    def _random_initial_parameters(self, annotations, estimate_omega):
        # TODO duplication w/ ModelBtLoopDesign
        if estimate_omega:
            # estimate omega from observed annotations
            omega = labels_frequency(annotations, self.nclasses)
        else:
            omega = self.omega

        theta = ModelA._random_theta(self.nannotators)
        return theta, omega


    ##### Model likelihood methods ############################################

    def log_likelihood(self, annotations):
        """Compute the log likelihood of a set of annotations given the model.

        Returns :math:`\log P(\mathbf{x} | \omega, \\theta)`,
        where :math:`\mathbf{x}` is the array of annotations.

        Arguments
        ---------
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

    # TODO code duplication with ModelBtLoopDesign -> refactor
    def _log_likelihood_counts(self, counts):
        """Compute the log likelihood of annotations given the model.

        This method assumes the data is in counts format.
        """

        # TODO: check if it's possible to replace these constraints with bounded optimization
        # check bounds of parameters (for likelihood optimization)
        if np.amin(self.theta) <= 0 or np.amax(self.theta) > 1:
            # return np.inf
            return SMALLEST_FLOAT

        # compute alpha and beta (they do not depend on theta)
        alpha = self._compute_alpha()
        beta = [None]*5
        pattern_to_indices = _compatibility_tables(self.nclasses)
        for pattern in range(5):
            indices = pattern_to_indices[pattern]
            beta[pattern] = self.omega[indices].prod(1)
            beta[pattern] /= beta[pattern].sum()

        llhood = 0.
        # loop over the 8 combinations of annotators
        for i in range(8):
            # extract the theta parameters for this triplet
            triplet_indices = np.arange(i, i+3) % self.nannotators
            triplet_indices.sort()
            theta_triplet = self.theta[triplet_indices]

            # compute the likelihood for the triplet
            llhood += self._log_likelihood_triplet(counts[:,i],
                                                   theta_triplet,
                                                   alpha, beta)

        return llhood


    def _log_likelihood_triplet(self, counts_triplet, theta_triplet,
                                alpha, beta):
        """Compute the log likelihood of data for one triplet of annotators.

        Input:
        counts_triplet -- count data for one combination of annotators
        theta_triplet -- theta parameters of the current triplet
        """

        nclasses = self.nclasses
        llhood = 0.

        # loop over all possible agreement patterns
        # 0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@

        pattern_to_indices = _compatibility_tables(nclasses)
        for pattern in range(5):
            # P( A_ijk | T_ijk ) * P( T_ijk )  , or "alpha * theta triplet"
            prob = self._prob_a_and_t(pattern, theta_triplet, alpha)

            # P( V_ijk ! A_ijk) * P( A_ijk | T_ijk ) * P( T_ijk )
            #   = P( V_ijk | A, T, model)
            prob *= beta[pattern]

            # P( V_ijk | model ) = sum over A and T of conditional probability
            indices = pattern_to_indices[pattern]
            count_indices = _triplet_to_counts_index(indices, nclasses)
            log_prob = ninf_to_num(np.log(prob))

            llhood += (counts_triplet[count_indices] * log_prob).sum()

        return llhood


    def _log_prior(self):
        """Compute log probability of prior on the theta parameters."""
        log_prob = scipy.stats.beta._logpdf(self.theta, 2., 1.).sum()
        if np.isneginf(log_prob):
            log_prob = SMALLEST_FLOAT
        return log_prob


    def _prob_a_and_t(self, pattern, theta_triplet, alpha):
        # TODO make more robust by taking logarithms earlier
        # TODO could be vectorized some more using the A_ijk tensor
        # 0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@

        # abbreviations
        thetat = theta_triplet
        not_thetat = (1.-theta_triplet)

        if pattern == 0:  # aaa patterns
            prob = (thetat.prod() + not_thetat.prod() * alpha[3])

        elif pattern == 1:  # aaA patterns
            prob = (thetat[0] * thetat[1] * not_thetat[2]
                    + not_thetat[0] * not_thetat[1] * thetat[2] * alpha[2]
                    + not_thetat[0] * not_thetat[1] * not_thetat[2] * alpha[4])

        elif pattern == 2:  # aAa patterns
            prob = (thetat[0] * not_thetat[1] * thetat[2]
                    + not_thetat[0] * thetat[1] * not_thetat[2] * alpha[1]
                    + not_thetat[0] * not_thetat[1] * not_thetat[2] * alpha[5])

        elif pattern == 3:  # Aaa patterns
            prob = (not_thetat[0] * thetat[1] * thetat[2]
                    + thetat[0] * not_thetat[1] * not_thetat[2] * alpha[0]
                    + not_thetat[0] * not_thetat[1] * not_thetat[2] * alpha[6])

        elif pattern == 4:  # Aa@ pattern
            prob = (not_thetat[0] * not_thetat[1] * not_thetat[2]
                      * (1. - alpha[3] - alpha[4] - alpha[5] - alpha[6])
                    + thetat[0] * not_thetat[1] * not_thetat[2]
                      * (1. - alpha[0])
                    + not_thetat[0] * thetat[1] * not_thetat[2]
                      * (1. - alpha[1])
                    + not_thetat[0] * not_thetat[1] * thetat[2]
                      * (1. - alpha[2]))

        return prob


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
        ---------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        nsamples : int
            number of samples to draw from the posterior

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
        save_params = self.theta
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
            self.theta = save_params


    ##### Posterior distributions #############################################

    # TODO ideally, one would infer the posterior over correctness (T_ijk)
    #   first, and then return the probability of each value
    #    def infer_correctness(self, annotations):
    #        """Infer posterior distribution over correctness patterns."""
    #        nitems = annotations.shape[0]
    #        nclasses = self.nclasses
    #
    #        posterior = np.zeros((nitems, self.annotators_per_item**2))
    #        alpha = self._compute_alpha()
    #        for i, row in enumerate(annotations):
    #            valid_idx = np.where(row >= 0)
    #            vijk = row[valid_idx]
    #            tijk = self.theta[valid_idx]
    #            p = self._compute_posterior_T_triplet(vijk, tijk, alpha)
    #            posteriors[i, :] = p
    #
    #        return posteriors
    #
    #
    #    def _compute_posterior_T_triplet(self, v, t, alpha):
    #        # switch over agreement pattern
    #        # 0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@
    #        if v[0] == v[1] == v[2]:  # aaa pattern
    #            pass


    def infer_labels(self, annotations):
        """Infer posterior distribution over label classes.

        Compute the posterior distribution over label classes given observed
        annotations, :math:`P( \mathbf{y} | \mathbf{x}, \\theta, \omega)`.

        Arguments
        ---------
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
        nclasses = self.nclasses

        posteriors = np.zeros((nitems, nclasses))
        alpha = self._compute_alpha()
        i = 0
        for row in annotations:
            ind = np.where(row >= 0)
            vijk = row[ind]
            tijk = self.theta[ind].copy()
            p = self._compute_posterior_triplet(vijk, tijk, alpha)
            posteriors[i, :] = p
            i += 1

        return posteriors


    def _compute_posterior_triplet(self, vijk, tijk, alpha):
        nclasses = self.nclasses
        posteriors = np.zeros(nclasses, float)

        #-----------------------------------------------
        # aaa
        if vijk[0] == vijk[1] and vijk[1] == vijk[2]:
            x1 = tijk[0] * tijk[1] * tijk[2]
            x2 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])
            p1 = x1 / (x1 + alpha[3] * x2)
            p2 = (1 - p1) / (nclasses - 1)

            for j in range(nclasses):
                if vijk[0] == j:
                    posteriors[j] = p1
                else:
                    posteriors[j] = p2

        #-----------------------------------------------
        # aaA
        elif vijk[0] == vijk[1] and vijk[1] != vijk[2]:
            x1 = tijk[0] * tijk[1] * (1 - tijk[2])
            x2 = (1 - tijk[0]) * (1 - tijk[1]) * tijk[2]
            x3 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

            # a is correct
            p1 = x1 / (x1 + alpha[2] * x2 + alpha[4] * x3)

            # A is correct
            p2 = (alpha[2] * x2) / (x1 + alpha[2] * x2 + alpha[4] * x3)

            # neither
            p3 = (1 - p1 - p2) / (nclasses - 2)

            for j in range(nclasses):
                if vijk[0] == j:
                    posteriors[j] = p1
                elif vijk[2] == j:
                    posteriors[j] = p2
                else:
                    posteriors[j] = p3

        #-----------------------------------------------
        # aAa
        elif vijk[0] == vijk[2] and vijk[1] != vijk[2]:
            x1 = tijk[0] * (1 - tijk[1]) * tijk[2]
            x2 = (1 - tijk[0]) * tijk[1] * (1 - tijk[2])
            x3 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

            # a is correct
            p1 = x1 / (x1 + alpha[1] * x2 + alpha[5] * x3)

            # A is correct
            p2 = (alpha[1] * x2) / (x1 + alpha[1] * x2 + alpha[5] * x3)

            # neither
            p3 = (1 - p1 - p2) / (nclasses - 2)

            for j in range(nclasses):
                if vijk[0] == j:
                    posteriors[j] = p1
                elif vijk[1] == j:
                    posteriors[j] = p2
                else:
                    posteriors[j] = p3

        #-----------------------------------------------
        # Aaa
        elif vijk[1] == vijk[2] and vijk[0] != vijk[2]:
            x1 = (1 - tijk[0]) * tijk[1] * tijk[2]
            x2 = tijk[0] * (1 - tijk[1]) * (1 - tijk[2])
            x3 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

            # a is correct
            p1 = x1 / (x1 + alpha[0] * x2 + alpha[6] * x3)

            # A is correct
            p2 = (alpha[0] * x2) / (x1 + alpha[0] * x2 + alpha[6] * x3)

            # neither
            p3 = (1 - p1 - p2) / (nclasses - 2)

            for j in range(nclasses):
                if vijk[0] == j:
                    posteriors[j] = p2
                elif vijk[2] == j:
                    posteriors[j] = p1
                else:
                    posteriors[j] = p3

        #-----------------------------------------------
        # aAb
        elif vijk[0] != vijk[1] and vijk[1] != vijk[2]:
            x1 = tijk[0] * (1 - tijk[1]) * (1 - tijk[2])
            x2 = (1 - tijk[0]) * tijk[1] * (1 - tijk[2])
            x3 = (1 - tijk[0]) * (1 - tijk[1]) * tijk[2]
            x4 = (1 - tijk[0]) * (1 - tijk[1]) * (1 - tijk[2])

            summa1 = 1 - alpha[3] - alpha[4] - alpha[5] - alpha[6]
            summa2 = ((1 - alpha[0]) * x1 + (1 - alpha[1]) * x2 +
                      (1 - alpha[2]) * x3 + summa1 * x4)

            # a is correct
            p1 = (1 - alpha[0]) * x1 / summa2

            # A is correct
            p2 = (1 - alpha[1]) * x2 / summa2

            # b is correct
            p3 = (1 - alpha[2]) * x3 / summa2

            # (a, A, b) are all incorrect
            p4 = (summa1 * x4 / summa2) / (nclasses - 3)

            for j in range(nclasses):
                if vijk[0] == j:
                    posteriors[j] = p1
                elif vijk[1] == j:
                    posteriors[j] = p2
                elif vijk[2] == j:
                    posteriors[j] = p3
                else:
                    posteriors[j] = p4

        # check posteriors: non-negative, sum to 1
        assert np.abs(posteriors.sum()-1.) < 1e-6
        assert posteriors.min() >= 0.

        return posteriors


    def _compute_alpha(self):
        """Compute the parameters `alpha` given the parameters `omega`.

        Cf. Table 4 in Rzhetsky et al., 2009.
        """

        omega = self.omega
        nclasses = self.nclasses
        alpha = np.zeros((7,))

        # ------ alpha_1,2,3

        # sum over all doublets
        outer_omega = np.outer(omega, omega)
        sum_wi_wk =  outer_omega.sum()

        # sum over all omega_i * omega_j, where i!=k and j!=k
        sum_wi_wj_not_k = np.zeros((nclasses,))
        # sum over all omega_i ** 2, where i!=k
        sum_wi2_not_k = np.zeros((nclasses,))

        for k in range(nclasses):
            sum_wi_wj_not_k[k] = (sum_wi_wk
                                  - 2*outer_omega[:,k].sum()
                                  + outer_omega[k,k])
            sum_wi2_not_k[k] = (outer_omega.diagonal().sum()
                                - outer_omega[k,k])

        a1 = (omega * sum_wi2_not_k / sum_wi_wj_not_k).sum()
        alpha[0:3] = a1

        # ------ alpha_4,5,6,7

        # sum over all triplets
        outer_omega3 = (outer_omega[:,:,np.newaxis]
                        * omega[np.newaxis,np.newaxis,:])
        sum_wi_wj_wl = outer_omega3.sum()

        # sum over omega_i * omega_j * omega_l, where i!=k and j!=k and l!=k
        sum_wi_wj_wl_not_k = np.zeros((nclasses,))
        for k in range(nclasses):
            sum_wi_wj_wl_not_k[k] = (sum_wi_wj_wl
                                     - 3.*outer_omega3[:,:,k].sum()
                                     + 3.*outer_omega3[:,k,k].sum()
                                     - outer_omega3[k,k,k])
        omega3 = omega**3
        sum_wi3_not_k = omega3.sum() - omega3

        a4 = (omega * sum_wi3_not_k / sum_wi_wj_wl_not_k).sum()
        alpha[3] = a4

        a5 = 0
        for i in range(nclasses):
            tmp = 0
            for j in range(nclasses):
                for k in range(nclasses):
                    if j != i and k != i and j != k:
                        tmp += omega[k] * omega[j] ** 2
            a5 += omega[i] * tmp / sum_wi_wj_wl_not_k[i]

        alpha[4:7] = a5

        return alpha


    ##### Verify input ########################################################

    def are_annotations_compatible(self, annotations):
        """Check if the annotations are compatible with the models' parameters.
        """

        if not super(ModelA, self).are_annotations_compatible(annotations):
            return False

        masked_annotations = np.ma.masked_equal(annotations, MISSING_VALUE)

        # exactly 3 annotations per row
        nvalid = (~masked_annotations.mask).sum(1)
        if not np.all(nvalid == self.nannotators_per_item):
            return False

        return True

