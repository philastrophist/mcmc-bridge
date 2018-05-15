import os
os.environ['MKL_THREADING_LAYER'] = "GNU"

from mcmc_bridge import EmceeTrace, export_to_emcee, get_start_point
from statsmodels.stats.moment_helpers import corr2cov

import numpy as np
import pymc3 as pm
import theano.tensor as tt
import theano


def read_test_data():
    """
    :return: Z vector array, S covariance matrix array
    """
    mu = np.arange(3)
    corr = np.array([-0.3, 0.1, 0.7])
    std = np.array([1, 2, 0.1])
    cov = corr2cov(corr, std)
    Z = np.random.multivariate_normal(mu, cov, 7000)
    S = np.tile(np.eye(3), (7000, 1, 1)) * 0.01
    return Z, S, (mu, corr, cov, std)


def correlation_model(values):
    """
    Builds Pymc3 correlation model for values without uncertainties.
    :param values: point vector of shape (ndata, ndimensions)
        :type values: np.array
    :return: Model
    :rtype: pm.Model
    """
    nmeas, ndim = values.shape
    with pm.Model() as model:
        nu = pm.HalfCauchy('nu', beta=10)  # prior on how much correlation (0 == rho=1, 1 == uniform prior on correlation, oo == no correlation)
        packed_L = pm.LKJCholeskyCov('packed_L', n=ndim, eta=nu, sd_dist=pm.HalfCauchy.dist(2.5))
        L = pm.expand_packed_triangular(ndim, packed_L)
        cov = pm.Deterministic('cov', L.dot(L.T))

        std = pm.Deterministic('std', tt.sqrt(tt.diag(cov)))
        diag = tt.diag(std)
        diaginv = tt.nlinalg.matrix_inverse(diag)
        corr = tt.dot(diaginv, tt.dot(cov, diaginv))

        indices = np.triu_indices(ndim, k=1)
        pm.Deterministic('corr_coeffs', corr[indices])

        mu = pm.Normal('mu', mu=values.mean(axis=0), sd=values.std(axis=0), shape=ndim)  # centre of gaussian
        pm.MvNormal('like', mu=mu, chol=L, observed=values, shape=ndim)
    return model


def test_model_backend():
    values = read_test_data()[0]
    with correlation_model(values) as model:
        sampler = export_to_emcee(nwalker_multiple=4)
        assert sampler.model is model
        start = get_start_point(sampler, n_init=1000)
        assert start.shape == (sampler.k, sampler.dim)
        nsteps = 33
        sampler.run_mcmc(start, nsteps)
        assert sampler.chain.shape == (sampler.k, nsteps, sampler.dim)
        trace = EmceeTrace(sampler)
        assert set(trace.varnames) == {i.name for i in model.unobserved_RVs}
        assert len(trace) == nsteps
        assert trace.nchains == sampler.k