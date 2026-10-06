"""
Monte Carlo samplers for the filtration scale-up model.

Normal and lognormal samples are set by the bounds of a confidence interval,
as in Guesstimate. Uniform samples are set by their end points.
"""

import numpy as np
from scipy.stats import norm


def sample_normal(low, high, n, confidence=90, absolute=False):
    """
    Generate random samples from a normal distribution. Based on Guesstimate's
    implementation, translated from Javascript to Python.

    Arguments:
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        n (int): The number of samples to generate.
        confidence (int): The confidence level. Must be 90, 95, or 99. Default 90.
        absolute (bool): If True, return absolute values of samples. Default False.

    Returns:
        numpy.ndarray: Random samples from the normal distribution, shape (n,).
    """
    if confidence == 90:
        z = 1.645
    elif confidence == 95:
        z = 1.96
    elif confidence == 99:
        z = 2.575
    else:
        raise ValueError("Confidence level must be 90, 95, or 99")

    mean = np.mean([high, low])
    stdev = (high - mean) / z
    samples = norm.rvs(loc=mean, scale=stdev, size=n)
    if absolute:
        samples = np.abs(samples)

    return samples


def sample_lognormal(low, high, n, confidence=90, absolute=False):
    """
    Generate random samples from a lognormal distribution.

    Arguments:
        low (float): The lower bound of the distribution. Must be > 0.
        high (float): The upper bound of the distribution.
        n (int): The number of samples to generate.
        confidence (int): The confidence level. Must be 90, 95, or 99. Default 90.
        absolute (bool): If True, return absolute values of samples. Default False.

    Returns:
        numpy.ndarray: Random samples from the lognormal distribution, shape (n,).
    """
    assert low > 0, "Low must be greater than 0 for lognormal distributions."

    if confidence == 90:
        z = 1.645
    elif confidence == 95:
        z = 1.96
    elif confidence == 99:
        z = 2.575
    else:
        raise ValueError("Confidence level must be 90, 95, or 99")
    logHigh = np.log(high)
    logLow = np.log(low)

    mean = np.mean([logHigh, logLow])
    stdev = (logHigh - logLow) / (2 * z)
    samples = np.random.lognormal(mean=mean, sigma=stdev, size=n)
    if absolute:
        samples = np.abs(samples)

    return samples


def sample_uniform(low, high, n):
    """
    Generate random samples from a uniform distribution.

    Arguments:
        low (float): The lower bound of the distribution.
        high (float): The upper bound of the distribution.
        n (int): The number of samples to generate.

    Returns:
        numpy.ndarray: Random samples from the uniform distribution, shape (n,).
    """
    return np.random.uniform(low=low, high=high, size=n)

