# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Andrey Rzhetsky
# License: Modified BSD license (2-clause)

"""This module defines functions to sample from a distribution given its
log likelihood.
"""

import numpy as np
from pyanno.util import PyannoValueError

import logging
logger = logging.getLogger(__name__)


def sample_distribution(likelihood, x0, arguments, step,
                        nsamples, x_lower, x_upper):
    """General-purpose sampling routine for MCMC sampling.

    Draw samples from a distribution given its unnormalized log likelihood
    using the Metropolis-Hasting Monte Carlo algorithm.

    It is recommended to optimize the step size, `step`, using the function
    :func:`optimize_step_size` in order to reduce the autocorrelation between
    successive samples.

    Arguments
    ---------
    likelihood : function(params, arguments)
        Function returning the *unnormalized* log likelihood of data given
        the parameters. The function accepts two arguments:
        `params` is the vector of parameters; `arguments` contains any
        additional argument that is needed to compute the log likelihood.

    x0 : ndarray, shape = (n_parameters, )
        Initial parameters value.

    arguments : any
        Additional argument passed to the function `likelihood`.

    step : ndarray, shape = (n_parameters, )
        Width of the proposal distribution over the parameters.

    nsamples : int
         Number of samples to draw from the distribution.

    x_lower : ndarray, shape = (n_parameters, )
        Lower bound for the parameters.

    x_upper : ndarray, shape = (n_parameters, )
        Upper bound for the parameters.
   """

    logger.info('Start collecting samples...')

    dim = len(x0)

    # array of samples
    samples = np.zeros((nsamples, dim))

    # current sample
    x_curr = x0.copy()

    # log likelihood of current sample
    llhood = likelihood(x0, arguments)

    for i in range(nsamples):
        if (i + 1) % 100 == 0:
            logger.info('... collected {} samples'.format(i+1))

        sum_log_q_ratio = 0.
        x_old = x_curr.copy()
        for j in range(dim):
            xj, log_q_ratio = sample_from_proposal_distribution(
                x_old[j], step[j], x_lower[j], x_upper[j]
            )
            sum_log_q_ratio += log_q_ratio
            x_curr[j] = xj

        llhood_new = likelihood(x_curr, arguments)

        # rejection step; reject with probability `alpha`
        alpha = min(1, np.exp(llhood_new - llhood + sum_log_q_ratio))
        if np.random.random() < alpha:
            # accept
            llhood = llhood_new
        else:
            # reject
            x_curr = x_old

        samples[i,:] = x_curr

    return samples


def optimize_step_size(likelihood, x0, arguments,
                       x_lower, x_upper,
                       n_samples,
                       recomputing_cycle, target_rejection_rate, tolerance):
    """Compute optimum jump for MCMC estimation of credible intervals.

    Adjust jump size in Metropolis-Hasting MC to achieve target rejection rate.
    Jump size is estimated for each parameter separately.

    Arguments
    ---------
    likelihood : function(params, arguments)
        Function returning the *unnormalized* log likelihood of data given
        the parameters. The function accepts two arguments:
        `params` is the vector of parameters; `arguments` contains any
        additional argument that is needed to compute the log likelihood.

    x0 : ndarray, shape = (n_parameters, )
        Initial parameters value.

    arguments : any
        Additional argument passed to the function `likelihood`.

    x_lower : ndarray, shape = (n_parameters, )
        Lower bound for the parameters.

    x_upper : ndarray, shape = (n_parameters, )
        Upper bound for the parameters.

    n_samples : int
        Total number of samples to draw during the optimization.

    recomputing_cycle : int
        Number of samples over which the rejection rates are computed. After
        `recomputing_cycle` samples, the step size is adapted and the
        rejection rates are reset to 0.

    target_rejection_rate : float
        Target rejection rate. If the rejection rate over the latest cycle is
        closer than `tolerance` to this target, the optimization phase is
        concluded.

    tolerance : float
        Tolerated deviation from `target_rejection_rate`.

    Returns
    -------
    step : ndarray, shape = (n_parameters, )
        The final optimized step size.
    """

    logger.info('Estimate optimal step size')

    dim = len(x0)

    # rejection rate for each paramter separately
    rejection_rate = np.zeros((dim,))

    # initial jump sizes
    step = (x_upper - x_lower) / 100.

    # *x_curr* is the current samples of the parameters
    x_curr = x0.copy()

    # log likelihood of current sample
    llhood = likelihood(x0, arguments)

    for i in range(n_samples):

        # we need to evaluate every dimension separately to have an estimate
        # of the rejection rate per parameter
        for j in range(dim):
            xj_old = x_curr[j]

            # draw new sample
            xj, log_q_ratio = sample_from_proposal_distribution(
                xj_old, step[j], x_lower[j], x_upper[j]
            )

            x_curr[j] = xj

            llhood_new = likelihood(x_curr, arguments)

            # rejection step; accept with probability `alpha`
            alpha = min(1, np.exp(llhood_new - llhood + log_q_ratio))

            if np.random.random() < alpha:
                llhood = llhood_new
            else:
                rejection_rate[j] += 1. / recomputing_cycle
                x_curr[j] = xj_old

        # adjust step size every `recomputing cycle` steps
        if i % recomputing_cycle == 0 and i > 0:
            logger.info('{} samples, adapt step size'.format(i))

            logger.debug('Rejection rate: ' + repr(rejection_rate))
            logger.debug('Step size: ' +repr(step))

            step, terminate = _adjust_jump(step,
                                           rejection_rate,
                                           target_rejection_rate,
                                           tolerance)

            if terminate == True:
                logger.debug('Step size within accepted range -- '
                             'exit optimization phase')
                break

            # reset all rejection rates
            rejection_rate *= 0.

    return step


# TODO: simplify, move stopping condition ("check") to optimize_step_size
def _adjust_jump(step, rejection_rate, target_reject_rate, tolerance):
    """Adapt step size to get closer to target rejection rate.

    Returns
    -------
    step : list
        new step sizes
    terminate : bool
        True if all rejection rates are within the required limits
    """

    terminate = True
    for j in range(step.shape[0]):
        step[j] = max(1e-6, step[j])

        if rejection_rate[j] != 0:
            if abs(rejection_rate[j] - target_reject_rate) > tolerance:
                step[j] *= (target_reject_rate / rejection_rate[j])
                terminate = False

        elif rejection_rate[j] == 0:
            step[j] *= 5.
            terminate = False

    return step, terminate


# TODO vectorize this to sample from all dimensions at once
def sample_from_proposal_distribution(theta, step, lower, upper):
    """Returns one sample from the proposal distribution.

    Arguments
    ---------
    theta : float
        current parameter value

    step : float
        width of the proposal distribution over `theta`

    lower : float
        lower bound for `theta`

    upper : float
        upper bound for `theta`

    Returns
    -------
    theta_new : float
        new sample from the distribution over theta

    log_q_ratio : float
        log-ratio of probability of new value given old value
        to probability of old value given new value
    """

    if theta < lower or theta > upper:
        raise PyannoValueError('Parameter values out or range')

    # *a* is a uniform random number
    a = np.random.random()

    # boundary conditions
    if theta == upper:
        theta = upper - a * min(step, upper - lower)
        q_new_to_old = -np.log(2 * min(step, upper - theta))
        q_old_to_new = -np.log(min(step, upper - lower))
        log_q_ratio = q_new_to_old - q_old_to_new
        return theta, log_q_ratio

    if theta == lower:
        theta = lower + a * min(step, upper - lower)
        q_new_to_old = -np.log(2. * min(step, theta - lower))
        q_old_to_new = -np.log(min(step, upper - lower))
        log_q_ratio = q_new_to_old - q_old_to_new
        return theta, log_q_ratio

    # go to the 'left'
    if  a > 0.5:
        theta_old = theta

        #jump interval is *theta*, choose uniformly
        theta -= np.random.random() * min(step, theta_old - lower)

        #transition probability from old to new
        q_old_to_new = -np.log(min(step, theta_old - lower))
        q_new_to_old = -np.log(min(step, upper - theta))
        log_q_ratio = q_new_to_old - q_old_to_new
        return theta, log_q_ratio

    # go to the 'right'
    else:
        # jump interval is *upper_limit*-*theta*, choose uniformly
        theta_old = theta
        theta += np.random.random() * min(step, upper - theta_old)

        q_old_to_new = -np.log(min(step, upper - theta_old))
        q_new_to_old = -np.log(min(step, theta - lower))
        log_q_ratio = q_new_to_old - q_old_to_new
        return theta, log_q_ratio
