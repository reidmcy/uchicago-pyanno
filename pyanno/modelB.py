# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Bob Carpenter
# License: Modified BSD license (2-clause)

"""This module defines the class ModelB, a Bayesian generalization
of the model proposed in (Dawid et al., 1979).

**Reference:**

* Dawid, A. P. and A. M. Skene. 1979.  Maximum likelihood
  estimation of observer error-rates using the EM algorithm.  Applied
  Statistics, 28(1):20--28.
"""

from traits.api import Int, Array
import numpy as np
from pyanno.abstract_model import AbstractModel
from pyanno.util import (random_categorical, create_band_matrix,
                         normalize, dirichlet_llhood,
                         is_valid, SMALLEST_FLOAT, PyannoValueError, labels_count)

import logging
logger = logging.getLogger(__name__)

ALPHA_DEFAULT = [16., 4., 2., 1.]


class ModelB(AbstractModel):
    """Bayesian generalization of the model proposed in (Dawid et al., 1979).

    Model B is a hierarchical generative model over annotations. The model
    assumes the existence of "true" underlying labels for each item,
    which are drawn from a categorical distribution,
    :math:`\pi`. Annotators report these labels with some noise, depending
    on their accuracy, :math:`\\theta`.

    These are the model parameters:

        - `pi[k]` is the probability of label k

        - `theta[j,k,k']` is the probability that annotator j reports label k'
          for an item whose real label is k, i.e.
          P( annotator j chooses k' | real label = k)

    The parameters are themselves random variables with these hyper-parameters:

        - `beta` are the parameters of the Dirichlet distribution over `pi`

        - `alpha[k,:]` are the parameters of the Dirichlet distributions over
          `theta[j,k,:]`

    See the documentation for a more detailed description of the model.

    **References:**

    * Dawid, A. P. and A. M. Skene. 1979.  Maximum likelihood
      estimation of observer error-rates using the EM algorithm.  Applied
      Statistics, 28(1):20--28.

    * Rzhetsky A., Shatkay, H., and Wilbur, W.J. (2009). "How to get the most
      from your curation effort", PLoS Computational Biology, 5(5).
    """


    ######## Model traits

    # number of label classes
    nclasses = Int

    # number of annotators
    nannotators = Int

    #### Model parameters

    # pi[k] is the prior probability of class k
    pi = Array(dtype=float, shape=(None,))

    # theta[j,k,:] is P(annotator j chooses : | real label = k)
    theta = Array(dtype=float, shape=(None, None, None))

    #### Hyperparameters

    # beta[:] are the parameters of the Dirichlet prior over pi[:]
    beta = Array(dtype=float, shape=(None,))

    # alpha[k,:] are the parameters of the Dirichlet prior over theta[j,k,:]
    alpha = Array(dtype=float, shape=(None, None))


    def __init__(self, nclasses, nannotators,
                 pi, theta,
                 alpha=None, beta=None, **traits):
        """Create an instance of ModelB.

        Arguments
        ----------
        nclasses : int
            Number of possible annotation classes

        nannotators : int
            Number of annotators

        pi : ndarray, shape = (n_classes,)
            pi[k] is the prior probability of class k.

        theta : ndarray, shape = (n_annotators, n_classes, n_classes)
            theta[j,k,k'] is the probability of annotator j reporting class k',
            while the true label is k.

        alpha : ndarray, shape = (n_classes, n_classes)
            Parameters of Dirichlet prior over annotator choices
            Default: peaks at correct annotation, decays to 1

        beta : ndarray
            Parameters of Dirichlet prior over model categories
            Default: beta[i] = 2.0
         """

        self.nclasses = nclasses
        self.nannotators = nannotators

        self.pi = pi
        self.theta = theta

        # initialize prior parameters if not specified
        if alpha is not None:
            self.alpha = alpha.copy()
        else:
            self.alpha = self.default_alpha(nclasses)

        if beta is not None:
            self.beta = beta.copy()
        else:
            self.beta = self.default_beta(nclasses)

        super(ModelB, self).__init__(**traits)


    ##### Model and data generation methods ###################################

    @staticmethod
    def create_initial_state(nclasses, nannotators, alpha=None, beta=None):
        """Factory method returning a model with random initial parameters.

        It is often more convenient to use this factory method over the
        constructor, as one does not need to specify the initial model
        parameters.

        The parameters theta and pi, controlling accuracy and prevalence,
        are initialized at random from the prior alpha and beta:

        :math:`\\theta_j^k \sim \mathrm{Dirichlet}(\mathbf{\\alpha_k})`

        :math:`\pi \sim \mathrm{Dirichlet}(\mathbf{\\beta})`

        If not defined, the prior parameters alpha ad beta are defined as
        described below.

        Arguments
        ---------
        nclasses : int
            Number of label classes

        nannotators : int
            Number of annotators

        alpha : ndarray
            Parameters of Dirichlet prior over annotator choices
            Default value is a band matrix that peaks at the correct
            annotation, with a value of 16 and decays to 1 with diverging
            classes. This prior is ideal for ordinal annotations.

        beta : ndarray
            Parameters of Dirichlet prior over model categories
            Default value for beta[i] is 1.0 .

        Returns
        -------
        model : :class:`~ModelB`
            Instance of ModelB
        """

        # NOTE: this is Bob Carpenter's prior; it is a *very* strong prior
        # over alpha for ordinal data, and becomes stronger for larger number
        # of classes. What is a sensible prior?
#        if alpha is None:
#            alpha = np.empty((nclasses, nclasses))
#            for k1 in xrange(nclasses):
#                for k2 in xrange(nclasses):
#                    # using Bob Carpenter's choice as a prior
#                    alpha[k1,k2] = max(1, (nclasses + (0.5 if k1 == k2 else 0)
#                                           - abs(k1 - k2)) ** 4)
#
#        if beta is None:
#            beta = 2.*np.ones(shape=(nclasses,))

        if alpha is None:
            alpha = ModelB.default_alpha(nclasses)

        if beta is None:
            beta = ModelB.default_beta(nclasses)

        # generate random distributions of prevalence and accuracy
        pi = np.random.dirichlet(beta)
        theta = ModelB._random_theta(nclasses, nannotators, alpha)

        return ModelB(nclasses, nannotators, pi, theta, alpha, beta)


    @staticmethod
    def _random_theta(nclasses, nannotators, alpha):
        theta = np.empty((nannotators, nclasses, nclasses))
        for j in range(nannotators):
            for k in range(nclasses):
                theta[j, k, :] = np.random.dirichlet(alpha[k, :])
        return theta


    @staticmethod
    def default_beta(nclasses):
        return np.ones((nclasses,))


    @staticmethod
    def default_alpha(nclasses):
        return create_band_matrix(nclasses, ALPHA_DEFAULT)


    def generate_labels(self, nitems):
        """Generate random labels from the model."""
        return random_categorical(self.pi, nitems)


    def generate_annotations_from_labels(self, labels):
        """Generate random annotations from the model, given labels

        The method samples random annotations from the conditional probability
        distribution of annotations, :math:`x_i^j`
        given labels, :math:`y_i`:

        :math:`x_i^j \sim \mathrm{Categorical}(\mathbf{\\theta_j^{y_i}})`

        Arguments
        ----------
        labels : ndarray, shape = (n_items,), dtype = int
            Set of "true" labels

        Returns
        -------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i
        """
        nitems = labels.shape[0]
        annotations = np.empty((nitems, self.nannotators), dtype=int)
        for j in range(self.nannotators):
            for i in range(nitems):
                annotations[i,j]  = (
                    random_categorical(self.theta[j,labels[i],:], 1))
        return annotations


    def generate_annotations(self, nitems):
        """Generate a random annotation set from the model.

        Sample a random set of annotations from the probability distribution
        defined the current model parameters:

            1) Label classes are generated from the prior distribution, pi

            2) Annotations are generated from the conditional distribution of
               annotations given classes, theta

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


    ##### Parameters estimation methods #######################################

    # TODO start from sample frequencies
    def map(self, annotations,
            epsilon=0.00001, init_accuracy=0.6, max_epochs=1000):
        """Computes maximum a posteriori (MAP) estimation of parameters.

        Estimate the parameters :attr:`theta` and :attr:`pi` from a set of
        observed annotations using maximum a posteriori estimation.

        Arguments
        ----------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        epsilon : float
            The estimation is interrupted when the objective function has
            changed less than `epsilon` on average over the last 10 iterations

        initial_accuracy : float
            Initialize the accuracy parameters, `theta` to a set of
            distributions where theta[j,k,k'] = initial_accuracy if k==k',
            and (1-initial_accuracy) / (n_classes - 1)

        max_epoch : int
            Interrupt the estimation after `max_epoch` iterations
        """

        self._raise_if_incompatible(annotations)

        map_em_generator = self._map_em_step(annotations, init_accuracy)
        self._parameter_estimation(map_em_generator, epsilon, max_epochs)


    def mle(self, annotations,
            epsilon=1e-5, init_accuracy=0.6, max_epochs=1000):
        """Computes maximum likelihood estimate (MLE) of parameters.

        Estimate the parameters :attr:`theta` and :attr:`pi` from a set of
        observed annotations using maximum likelihood estimation.

        Arguments
        ----------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        epsilon : float
            The estimation is interrupted when the objective function has
            changed less than `epsilon` on average over the last 10 iterations

        initial_accuracy : float
            Initialize the accuracy parameters, `theta` to a set of
            distributions where theta[j,k,k'] = initial_accuracy if k==k',
            and (1-initial_accuracy) / (n_classes - 1)

        max_epoch : int
            Interrupt the estimation after `max_epoch` iterations
        """

        self._raise_if_incompatible(annotations)

        mle_em_generator = self._mle_em_step(annotations, init_accuracy)
        self._parameter_estimation(mle_em_generator, epsilon, max_epochs)


    def _parameter_estimation(self, learning_iterator, epsilon, max_epochs):

        if epsilon < 0.0: raise PyannoValueError("epsilon < 0.0")
        if max_epochs < 0: raise PyannoValueError("max_epochs < 0")

        info_str = "Epoch={0:6d}  obj={1:+10.4f}   diff={2:10.4f}"

        logger.info('Start parameters optimization...')

        epoch = 0
        obj_history = []
        diff = np.inf
        for objective, prev_est, cat_est, acc_est in learning_iterator:
            logger.info(info_str.format(epoch, objective, diff))

            obj_history.append(objective)

            # stopping conditions
            if epoch > max_epochs: break
            if epoch > 10:
                diff = (obj_history[epoch] - obj_history[epoch-10]) / 10.0
                if abs(diff) < epsilon: break

            epoch += 1

        logger.info('Parameters optimization finished')

        # update internal parameters
        self.pi = prev_est
        self.theta = acc_est

        return cat_est


    def _map_em_step(self, annotations, init_accuracy=0.6):
       # TODO move argument checking to traits
#        if not np.all(beta > 0.):
#            raise ValueError("beta should be larger than 0")
#        if not np.all(alpha > 0.):
#            raise ValueError("alpha should be larger than 0")
#
#        if annotations.shape != (nitems, nannotators):
#            raise ValueError("size of `annotations` should be nitems x nannotators")
#        if init_accuracy < 0.0 or init_accuracy > 1.0:
#            raise ValueError("init_accuracy not in [0,1]")
#        if len(alpha) != nclasses:
#            raise ValueError("len(alpha) != K")
#        for k in xrange(nclasses):
#            if len(alpha[k]) != nclasses:
#                raise ValueError("len(alpha[k]) != K")
#        if len(beta) != nclasses:
#            raise ValueError("len(beta) != K")

        # True if annotations is missing
        missing_mask_nclasses = self._missing_mask(annotations)

        # prevalence is P( category )
        prevalence = self._compute_prevalence()
        accuracy = self._initial_accuracy(init_accuracy)

        while True:
            # Expectation step (E-step)
            # compute marginal likelihood P(category[i] | model, data)

            log_likelihood, unnorm_category = (
                self._log_likelihood_core(annotations,
                                          prevalence,
                                          accuracy,
                                          missing_mask_nclasses)
            )
            log_prior = self._log_prior(prevalence, accuracy)

            # category is P(category[i] = k | model, data)
            category = unnorm_category / unnorm_category.sum(1)[:,None]

            # return here with E[cat|prev,acc] and LL(prev,acc;y)
            yield (log_prior+log_likelihood, prevalence, category, accuracy)

            # Maximization step (M-step)
            # update parameters to maximize likelihood
            prevalence = self._compute_prevalence(category)
            accuracy = self._compute_accuracy(category, annotations,
                                              use_prior=True)


    def _mle_em_step(self, annotations, init_accuracy=0.6):
        # True if annotations is missing
        missing_mask_nclasses = self._missing_mask(annotations)

        # prevalence is P( category )
        prevalence = np.empty((self.nclasses,))
        prevalence.fill(1. / float(self.nclasses))
        accuracy = self._initial_accuracy(init_accuracy)

        while True:
            # Expectation step (E-step)
            # compute marginal likelihood P(category[i] | model, data)

            log_likelihood, unnorm_category = (
                self._log_likelihood_core(annotations,
                                          prevalence,
                                          accuracy,
                                          missing_mask_nclasses)
            )

            # category is P(category[i] = k | model, data)
            category = unnorm_category / unnorm_category.sum(1)[:,None]

            # return here with E[cat|prev,acc] and LL(prev,acc;y)
            yield (log_likelihood, prevalence, category, accuracy)

            # Maximization step (M-step)
            # update parameters to maximize likelihood
            prevalence = normalize(category.sum(0))
            accuracy = self._compute_accuracy(category, annotations,
                                              use_prior=False)


    def _compute_prevalence(self, category=None):
        """Return prevalence, P( category )."""
        beta_prior_count = self.beta - 1.
        if category is None:
            # initialize at the *mode* of the distribution
            return normalize(beta_prior_count)
        else:
            return normalize(beta_prior_count + category.sum(0))


    def _initial_accuracy(self, init_accuracy):
        """Return initial setting for accuracy."""
        nannotators = self.nannotators
        nclasses = self.nclasses

        # accuracy[j,k,k'] is P(annotation_j = k' | category=k)
        accuracy = np.empty((nannotators, nclasses, nclasses))
        accuracy.fill((1. - init_accuracy) / (nclasses - 1.))
        for k in range(nclasses):
            accuracy[:, k, k] = init_accuracy
        return accuracy


    def _compute_accuracy(self, category, annotations, use_prior):
        """Return accuracy, P(annotation_j = k' | category=k)

        Helper function to compute an estimate of the accuracy parameters
        theta, given labels and annotations.

        Returns
        -------
        accuracy : ndarray, shape = (n_annotators, n_classes, n_classes)
            accuracy[j,k,k'] = P(annotation_j = k' | category=k).
        """
        nitems, nannotators = annotations.shape
        # alpha - 1 : the mode of a Dirichlet is  (alpha_i - 1) / (alpha_0 - K)
        alpha_prior_count = self.alpha - 1.
        valid_mask = is_valid(annotations)

        annotators = np.arange(nannotators)[None,:]
        if use_prior:
            accuracy = np.tile(alpha_prior_count, (nannotators, 1, 1))
        else:
            accuracy = np.zeros((nannotators, self.nclasses, self.nclasses))

        for i in range(nitems):
            valid = valid_mask[i,:]
            accuracy[annotators[:,valid],:,annotations[i,valid]] += category[i,:]
        accuracy /= accuracy.sum(2)[:, :, None]

        return accuracy


    def _compute_category(self, annotations, prevalence, accuracy,
                          missing_mask_nclasses=None, normalize=True):
        """Compute P(category[i] = k | model, annotations).

        Arguments
        ----------
        annotations : ndarray
            Array of annotations

        prevalence : ndarray
            Gamma parameters

        accuracy : ndarray
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

        # unnorm_category is P(category[i] = k | model, data), unnormalized
        unnorm_category = np.tile(prevalence.copy(), (nitems, 1))
        # mask missing annotations
        annotators = np.arange(nannotators)[None, :]
        tmp = np.ma.masked_array(accuracy[annotators, :, annotations],
                                 mask=missing_mask_nclasses)
        unnorm_category *= tmp.prod(1)

        if normalize:
            return unnorm_category / unnorm_category.sum(1)[:,None]

        return unnorm_category


    def _missing_mask(self, annotations):
        missing_mask = ~ is_valid(annotations)
        missing_mask_nclasses = np.tile(missing_mask[:, :, None],
            (1, 1, self.nclasses))
        return missing_mask_nclasses


    ##### Model likelihood methods ############################################

    def log_likelihood(self, annotations):
        """Compute the log likelihood of a set of annotations given the model.

        Returns log P(annotations | current model parameters).

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

        missing_mask_nclasses = self._missing_mask(annotations)
        llhood, _ = self._log_likelihood_core(annotations,
                                              self.pi, self.theta,
                                              missing_mask_nclasses)
        return llhood


    def _log_likelihood_core(self, annotations,
                             prevalence, accuracy,
                             missing_mask_nclasses):

        unnorm_category = self._compute_category(annotations,
                                                 prevalence, accuracy,
                                                 missing_mask_nclasses,
                                                 normalize=False)

        llhood = np.log(unnorm_category.sum(1)).sum()
        if np.isnan(llhood):
            llhood = SMALLEST_FLOAT

        return llhood, unnorm_category


    def _log_prior(self, prevalence, accuracy):
        alpha = self.alpha
        log_prior = dirichlet_llhood(prevalence, self.beta)
        for j in range(self.nannotators):
            for k in range(self.nclasses):
                log_prior += dirichlet_llhood(accuracy[j,k,:], alpha[k])
        return log_prior


    ##### Sampling posterior over parameters ##################################

    def sample_posterior_over_accuracy(self, annotations, nsamples,
                                       burn_in_samples=0,
                                       thin_samples=1,
                                       return_all_samples=True):
        """Return samples from posterior distribution over theta given data.

        Samples are drawn using Gibbs sampling, i.e., alternating between
        sampling from the conditional distribution of theta given the
        annotations and the label classes, and sampling from the conditional
        distribution of the classes given theta and the annotations.

        This results in a fast-mixing sampler, and so the parameters
        controlling burn-in and thinning can be set to a small number
        of samples.

        Arguments
        ----------
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

        return_all_samples : bool
            If True, return not only samples for the parameters theta,
            but also for the parameters pi, and the label classes, y.

        Returns
        -------
        samples : ndarray, shape = (n_samples, n_annotators, nclasses, nclasses)
            samples[i,...] is one sample from the posterior distribution over
            the parameters `theta`

        (theta, pi, labels) : tuple of ndarray
            If the keyword argument `return_all_samples` is set to True,
            return a tuple with the samples for the parameters theta,
            the parameters pi, and the label classes, y
        """

        self._raise_if_incompatible(annotations)
        nsamples = self._compute_total_nsamples(nsamples,
                                                burn_in_samples,
                                                thin_samples)

        logger.info('Start collecting samples...')

        # use Gibbs sampling
        nitems, nannotators = annotations.shape
        nclasses = self.nclasses
        alpha_prior = self.alpha
        beta_prior = self.beta

        # arrays holding the current samples
        theta_samples = np.empty((nsamples, nannotators, nclasses, nclasses))
        if return_all_samples:
            pi_samples = np.empty((nsamples, nclasses))
            label_samples = np.empty((nsamples, nitems))

        theta_curr = self.theta.copy()
        pi_curr = self.pi.copy()
        label_curr = np.empty((nitems,), dtype=int)

        for sidx in range(nsamples):
            if ((sidx+1) % 50) == 0:
                logger.info('... collected {} samples'.format(sidx+1))

            ####### A: sample labels given annotations, theta and pi

            ### compute posterior over label classes given theta and pi
            category_distr = self._compute_category(annotations,
                                                    pi_curr,
                                                    theta_curr)

            ### sample from the categorical distribution over labels
            # 1) precompute cumulative distributions
            cum_distr = category_distr.cumsum(1)

            # 2) precompute random values
            rand = np.random.random(nitems)
            for i in range(nitems):
                # 3) samples from i-th categorical distribution
                label_curr[i] = cum_distr[i,:].searchsorted(rand[i])

            ####### B: sample theta given annotations and label classes

            # 1) compute alpha parameters of Dirichlet posterior
            alpha_post = np.tile(alpha_prior, (nannotators, 1, 1))
            for l in range(nannotators):
                for k in range(nclasses):
                    for i in range(nclasses):
                        alpha_post[l,k,i] += ((label_curr==k)
                                              & (annotations[:,l]==i)).sum()

            # 2) sample thetas
            for l in range(nannotators):
                for k in range(nclasses):
                    theta_curr[l,k,:] = np.random.dirichlet(alpha_post[l,k,:])

            ####### C: sample pi given label classes

            # 1) compute beta parameters of dirichlet posterior
            # number of labels of class k
            count = np.bincount(label_curr, minlength=nclasses)
            beta_hat = beta_prior + count
            pi_curr = np.random.dirichlet(beta_hat)

            # copy current samples
            theta_samples[sidx,...] = theta_curr
            if return_all_samples:
                pi_samples[sidx,:] = pi_curr
                label_samples[sidx,:] = label_curr

        theta_samples = self._post_process_samples(theta_samples,
                                                   burn_in_samples,
                                                   thin_samples)
        if return_all_samples:
            pi_samples = self._post_process_samples(pi_samples,
                                                    burn_in_samples,
                                                    thin_samples)
            label_samples = self._post_process_samples(label_samples,
                                                       burn_in_samples,
                                                       thin_samples)
            return (theta_samples, pi_samples, label_samples)

        return theta_samples


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
                                          self.pi,
                                          self.theta)

        return category


    ##### Compute accuracy ###################################################

    def annotator_accuracy(self):
        """Return the accuracy of each annotator.

        Compute a summary of the a-priori accuracy of each annotator, i.e.,
        P( annotator j is correct ). This can be computed from the parameters
        theta and pi, as

        P( annotator j is correct )
        = \sum_k P( annotator j reports k | label is k ) P( label is k )
        = \sum_k theta[j,k,k] * pi[k]

        Returns
        -------
        accuracy : ndarray, shape = (n_annotators, )
            accuracy[j] = P( annotator j is correct )
        """

        accuracy = np.zeros((self.nannotators,))
        for k in range(self.nclasses):
            accuracy += self.theta[:,k,k] * self.pi[k]
        return accuracy


    def annotator_accuracy_samples(self, theta_samples, pi_samples):
        """Return samples from the accuracy of each annotator.

        Given samples from the posterior of accuracy parameters theta
        (see :method:`sample_posterior_over_accuracy`), compute
        samples from the posterior distribution of the annotator accuracy,
        i.e.,

        P( annotator j is correct | annotations).

        See also :method:`sample_posterior_over_accuracy`,
        :method:`annotator_accuracy`

        Returns
        -------
        accuracy : ndarray, shape = (n_annotators, )
            accuracy[j] = P( annotator j is correct )
        """

        nsamples = theta_samples.shape[0]

        accuracy = np.zeros((nsamples, self.nannotators,))
        for k in range(self.nclasses):
            accuracy += theta_samples[:,:,k,k] * pi_samples[:,k,np.newaxis]
        return accuracy
