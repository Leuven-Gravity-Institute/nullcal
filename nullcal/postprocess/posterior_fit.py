import bilby
import numpy as np

def fit_gaussian_posterior_from_result(result):
    samples = result.samples
    nsample, ndim = samples.shape
    mu = np.mean(samples, axis=0)
    cov = np.cov(samples.T)
    return bilby.core.prior.MultivariateGaussianDist(names=list(result.posterior.columns[:-2]), mus=mu, covs=cov)