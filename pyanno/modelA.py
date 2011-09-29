"""Definition of model A."""

from collections import defaultdict
import numpy as np
import scipy.optimize
from pyanno.sampling import optimum_jump, sample_distribution
from pyanno.util import compute_counts, random_categorical, log_beta_pdf


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

    if not _compatibility_tables_cache.has_key(nclasses):
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
    return (triplet * np.array([nclasses**2, nclasses, 1])).sum(1)


class ModelA(object):
    """
    At the moment the model assumes 1) a total of 8 annotators, and 2) each
    item is annotated by 3 annotators.
    """

    def __init__(self, nclasses, theta, omega):
        self.nclasses = nclasses
        self.nannotators = 8
        # number of annotators rating each item in the loop design
        self.annotators_per_item = 3
        self.theta = theta
        self.omega = omega


    # ---- model and data generation methods

    @staticmethod
    def random_model(nclasses, theta=None, omega=None):
        """Factory method returning a random model.

         Input:
         nclasses -- number of categories
         theta -- probability of annotators being correct
         alpha -- probability of drawing agreement pattern given
                  correctness pattern
         omega -- probability of drawing an annotation value
                  (assumed to be the same for all annotators)
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
        """Generate random annotations given labels."""
        theta = self.theta
        nannotators = self.nannotators
        nitems_per_loop = nitems // nannotators

        annotations = np.empty((nitems, nannotators), dtype=int)
        annotations.fill(-1)

        # loop over annotator triplets (loop design)
        for j in range(nannotators):
            triplet_indices = np.arange(j, j+3) % self.nannotators

            # -- step 1: generate correct / incorrect labels

            # parameters for this triplet
            theta_triplet = self.theta[triplet_indices]
            incorrect = self._generate_incorrectness(nitems_per_loop,
                                                     theta_triplet)

            # -- step 2: generate agreement patterns given correctness
            # convert boolean correctness into combination indices
            # (indices as in Table 3)
            agreement = self._generate_agreement(incorrect)

            # -- step 3: generate annotations
            annotations[j*nitems_per_loop:(j+1)*nitems_per_loop,
            triplet_indices] = (
                self._generate_annotations(agreement)
                )

        return annotations


    def _generate_incorrectness(self, n, theta_triplet):
        _rnd = np.random.rand(n, self.annotators_per_item)
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
        for i in xrange(nitems_per_loop):
            # generate agreement pattern according to A_ijk
            agreement[i] = random_categorical(
                agreement_tbl[correct_idx[i]], 1)

        return agreement


    def _generate_annotations(self, agreement):
        """Generate triplet annotations given agreement pattern."""
        nitems_per_loop = agreement.shape[0]
        omega = self.omega
        annotations = np.empty((nitems_per_loop, 3), dtype=int)

        for i in xrange(nitems_per_loop):
            # get all compatible annotations
            compatible = _compatibility_tables(self.nclasses)[agreement[i]]
            # compute probability of each possible annotation
            distr = omega[compatible].prod(1)
            distr /= distr.sum()
            # draw annotation
            compatible_idx = random_categorical(distr, 1)[0]
            annotations[i,:] = compatible[compatible_idx, :]
        return annotations


    # ---- Parameters estimation methods

    def mle(self, annotations, estimate_alpha=False, estimate_omega=True,
            use_prior = True):
        counts = compute_counts(annotations, self.nclasses)

        def _wrap_lhood(params):
            self.theta = params
            return - self._log_likelihood_counts(counts)

        params_start, omega = self._random_initial_parameters(annotations,
                                                              estimate_omega)
        self.omega = omega
        params_best = scipy.optimize.fmin(_wrap_lhood,
                                          params_start,
                                          xtol=1e-4, ftol=1e-4, disp=True,
                                          maxiter=1e+10, maxfun=1e+30)

        if estimate_alpha:
            self.alpha[:3] = params_best[8]
            self.alpha[3] = params_best[9]
            self.alpha[4:] = params_best[10]
        self.theta = params_best[:8]


    def _random_initial_parameters(self, annotations, estimate_omega):
        # TODO duplication w/ ModelBt
        if estimate_omega:
            # estimate gamma from observed annotations
            omega = (np.bincount(annotations[annotations!=-1])
                          / (3.*annotations.shape[0]))
        else:
            omega = ModelA._random_omega(self.nclasses)

        theta = ModelA._random_theta(self.nannotators)
        return theta, omega


    # ---- model likelihood

    # TODO code duplication with ModelBt -> refactor
    def log_likelihood(self, annotations, use_prior=False):
        """Compute the log likelihood of annotations given the model."""
        return self._log_likelihood_counts(compute_counts(annotations,
                                                          self.nclasses),
                                           use_prior)


    def _log_likelihood_counts(self, counts, use_prior=False):
        """Compute the log likelihood of annotations given the model.

        This method assumes the data is in counts format.
        """

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
                                                   alpha, beta, use_prior)

        return llhood


    def _log_likelihood_triplet(self, counts_triplet, theta_triplet,
                                alpha, beta, use_prior):
        """Compute the log likelihood of data for one triplet of annotators.

        Input:
        counts_triplet -- count data for one combination of annotators
        theta_triplet -- theta parameters of the current triplet
        """

        nclasses = self.nclasses
        omega = self.omega
        llhood = 0.

        # TODO: check if it's possible to replace these constraints with bounded optimization
        # TODO: this check can be done in _log_likelihood_counts
        if np.amin(theta_triplet) <= 0 or np.amax(theta_triplet) > 1:
            # return np.inf
            return -1e20

        # prior on *thetas*
        if use_prior:
            llhood += log_beta_pdf(theta_triplet, 2., 1.).sum()

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
            llhood += (counts_triplet[count_indices] * np.log(prob)).sum()

        return llhood


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
            prob = (  thetat[0] * thetat[1] * not_thetat[2]
                      + not_thetat[0] * not_thetat[1] * thetat[2] * alpha[2]
                      + not_thetat[0] * not_thetat[1] * not_thetat[2] * alpha[
                                                                        4])

        elif pattern == 2:  # aAa patterns
            prob = (  thetat[0] * not_thetat[1] * thetat[2]
                      + not_thetat[0] * thetat[1] * not_thetat[2] * alpha[1]
                      + not_thetat[0] * not_thetat[1] * not_thetat[2] * alpha[
                                                                        5])

        elif pattern == 3:  # Aaa patterns
            prob = (  not_thetat[0] * thetat[1] * thetat[2]
                      + thetat[0] * not_thetat[1] * not_thetat[2] * alpha[0]
                      + not_thetat[0] * not_thetat[1] * not_thetat[2] * alpha[
                                                                        6])

        elif pattern == 4:  # Aa@ pattern
            prob = (  not_thetat[0] * not_thetat[1] * not_thetat[2]
                      * (1. - alpha[3] - alpha[4] - alpha[5] - alpha[6])
                      + thetat[0] * not_thetat[1] * not_thetat[2]
            * (1. - alpha[0])
                      + not_thetat[0] * thetat[1] * not_thetat[2]
            * (1. - alpha[1])
                      + not_thetat[0] * not_thetat[1] * thetat[2]
            * (1. - alpha[2]))

        return prob


    # ---- sampling posterior over parameters

    # TODO duplicated functionality w/ ModelBt, refactor
    # TODO arguments for burn-in, thinning
    def sample_posterior_over_theta(self, annotations, nsamples,
                                    target_rejection_rate = 0.3,
                                    rejection_rate_tolerance = 0.05,
                                    step_optimization_nsamples = 500,
                                    adjust_step_every = 100):
        """Return samples from posterior distribution over theta given data.
        """
        # optimize step size
        counts = compute_counts(annotations, self.nclasses)

        # wrap log likelihood function to give it to optimum_jump and
        # sample_distribution
        _llhood_counts = self._log_likelihood_counts
        def _wrap_llhood(params, counts):
            self.theta = params
            # minimize *negative* likelihood
            return _llhood_counts(counts)

        # TODO this save-reset is rather ugly, refactor: create copy of
        #      model and sample over it
        # save internal parameters to reset at the end of sampling
        save_params = self.theta
        try:
            # compute optimal step size for given target rejection rate
            params_start = self.theta.copy()
            params_upper = np.ones((self.nannotators,))
            params_lower = np.zeros((self.nannotators,))
            step = optimum_jump(_wrap_llhood, params_start, counts,
                                params_upper, params_lower,
                                step_optimization_nsamples,
                                adjust_step_every,
                                target_rejection_rate,
                                rejection_rate_tolerance, 'Everything')

            # draw samples from posterior distribution over theta
            samples = sample_distribution(_wrap_llhood, params_start, counts,
                                          step, nsamples,
                                          params_lower, params_upper,
                                          'Everything')
            return samples
        finally:
            # reset parameters
            self.theta = save_params


    def infer_labels(self, annotations):
        """Infer posterior distribution over true labels."""
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
            summa2 = (1 - alpha[0]) * x1 + (1 - alpha[1]) * x2 + (1 - alpha[
                                                                        2]) * x3 + summa1 * x4

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
        if sum(posteriors) - 1.0 > 0.0000001 or min(posteriors) < 0:
            print 'Aberrant posteriors!!!', posteriors
            print sum(posteriors)
            print min(posteriors)
            time.sleep(60)

        return posteriors


    # TODO optimize using outer products
    def _compute_alpha(self):
        """Compute the parameters `alpha` given the parameters `omega`."""

        omega = self.omega
        nclasses = self.nclasses
        alpha = np.zeros((7,))

        # dublet sums
        s2 = np.zeros((nclasses,))
        s3 = np.zeros((nclasses,))

        for k in range(nclasses):
            for i in range(nclasses):
                for j in range(nclasses):
                    if i != k and j != k:
                        s2[k] += omega[i] * omega[j]
                        # triplet sums

        for k in range(nclasses):
            for i in range(nclasses):
                for j in range(nclasses):
                    for l in range(nclasses):
                        if i != k and j != k and l != k:
                            s3[k] += omega[i] * omega[j] * omega[l]

        # a1 / a2 / a3
        a1 = 0
        for i in range(nclasses):
            tmp = 0
            for j in range(nclasses):
                if j != i:
                    tmp += omega[j] ** 2
            a1 += omega[i] * tmp / s2[i]

        alpha[0] = a1
        alpha[1] = a1
        alpha[2] = a1

        # a4

        a4 = 0
        for i in range(nclasses):
            tmp = 0
            for j in range(nclasses):
                if j != i:
                    tmp += omega[j] ** 3
            a4 += omega[i] * tmp / s3[i]

        alpha[3] = a4

        # a5, a6, a7

        a5 = 0
        for i in range(nclasses):
            tmp = 0
            for j in range(nclasses):
                for k in range(nclasses):
                    if j != i and k != i and j != k:
                        tmp += omega[k] * omega[j] ** 2
            a5 += omega[i] * tmp / s3[i]

        alpha[4] = a5
        alpha[5] = a5
        alpha[6] = a5

        return alpha