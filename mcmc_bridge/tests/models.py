import pymc3 as pm
import numpy as np
from statsmodels.stats.moment_helpers import corr2cov
from theano import tensor as tt


def linear():
    np.random.seed(0)
    size = 200
    true_intercept = 1
    true_slope = 2

    x = np.linspace(0, 1, size)
    # y = a + b*x
    true_regression_line = true_intercept + true_slope * x
    # add noise
    y = true_regression_line + np.random.normal(scale=.1, size=size)

    data = dict(x=x, y=y)

    with pm.Model() as model:  # model specifications in PyMC3 are wrapped in a with-statement
        pm.glm.GLM.from_formula('y ~ x', data)
    return model, {'Intercept': (true_intercept, 0.1), 'x': (true_slope, 0.1)}


def read_test_data():
    """
    :return: Z vector array, S covariance matrix array
    """
    mu = np.arange(3)
    corr_coeffs = np.array([-0.3, 0.1, 0.7])
    ndim = int(np.sqrt(1 + (4 * len(corr_coeffs) * 2)) + 1) // 2
    corr = np.zeros((ndim, ndim))
    corr[np.triu_indices(ndim, k=1)] = corr_coeffs
    corr[np.diag_indices(ndim)] = 1
    corr = corr + corr.T - np.diag(corr.diagonal())

    std = np.array([1, 2, 0.1])
    cov = np.dot(corr, np.dot(np.diag(std), corr))
    Z = np.random.multivariate_normal(mu, cov, 7000)
    S = np.tile(np.eye(3), (7000, 1, 1)) * 0.01
    return Z, S, (mu, corr_coeffs, corr, cov, std)


def correlation():
    np.random.seed(0)
    values, S, (true_mu, true_corr_coeffs, true_corr, true_cov, true_std) = read_test_data()
    nmeas, ndim = values.shape
    with pm.Model() as model:
        packed_L = pm.LKJCholeskyCov('packed_L', n=ndim, eta=1, sd_dist=pm.HalfCauchy.dist(2.5))
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
    return model, {'mu': (true_mu, 0.1), 'cov': (true_cov, 0.1)}


