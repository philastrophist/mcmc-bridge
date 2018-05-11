import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from pymc3.backends.base import MultiTrace
from pymc3.backends.ndarray import NDArray
from tqdm import tqdm

theano.config.compute_test_value = "ignore"


def get_unfitted_parameters(sampler, output_vars, model=None):
        model = pm.modelcontext(model)
        fn = model.fastfn(output_vars)
        for i, sample in enumerate(sampler.flatchain.T):
            point = array2point(sample, model=model)
            results = fn(point)
            if i == 0:
                output_arrays = [np.zeros(sampler.flatchain.shape[:1] + r.shape) for r in results]
            for j, r in enumerate(results):
                output_arrays[j][i] = r
        return results


def get_scalar_loglikelihood_function(model, vars=None, nans='-inf'):
    assert nans == '-inf' or nans == 'raise', "nans must be either 'raise' or '-inf'"
    model = pm.modelcontext(model)
    if vars is None:
        vars = model.vars
    shared = pm.make_shared_replacements(vars, model)
    [logp0], inarray0 = pm.join_nonshared_inputs([model.logpt], vars, shared)

    f = theano.function([inarray0], logp0)
    f.trust_input = True

    unobserved = list(set(model.unobserved_RVs) - set(model.vars))
    supplementary_fn = model.fastfn(unobserved)

    def _lnpost(theta):
        v = f(theta)
        supplementary = supplementary_fn(array2point(theta))
        if nans == '-inf' and np.isnan(v):
            return -np.inf, supplementary
        return v, supplementary

    return _lnpost, [u.name for u in unobserved]


def export_to_emcee(model=None, nwalker_multiple=2, **kwargs):
    import emcee
    model = pm.modelcontext(model)
    lnpost, unobserved_varnames = get_scalar_loglikelihood_function(model)
    dim = sum(var.dsize for var in model.vars)
    sampler = emcee.EnsembleSampler(nwalker_multiple*dim, dim, lnpost, **kwargs)
    sampler.model = model
    sampler.unobserved_varnames = unobserved_varnames
    sampler.ordering = pm.ArrayOrdering(model.vars)
    return sampler


def point2array(point, model=None, vars=None):
    model = pm.modelcontext(model)
    if vars is None:
        vars = model.vars
    ordering = pm.ArrayOrdering(vars)
    shared = pm.make_shared_replacements(vars, model)
    shared = {str(var): shared for var, shared in shared.items()}

    for var, share in shared.items():
        share.set_value(point[var])

    bij = pm.DictToArrayBijection(ordering, point)
    return bij.map(point)


def array2point(apoint, model=None, vars=None):
    model = pm.modelcontext(model)
    if vars is None:
        vars = model.vars
    ordering = pm.ArrayOrdering(vars)

    bij = pm.DictToArrayBijection(ordering, model.test_point)
    return bij.rmap(apoint)


def get_nwalkers(sampler):
    try:
        return sampler.nwalkers
    except AttributeError:
        return sampler.k


def get_start_point(sampler, init='advi', n_init=500000, progressbar=True, **kwargs):
    start, _ = pm.init_nuts(init, get_nwalkers(sampler), n_init=n_init, model=sampler.model, progressbar=progressbar, **kwargs)
    return np.asarray([point2array(s, sampler.model, sampler.model.vars) for s in start])


if __name__ == '__main__':
    import numpy as np
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
        # x = pm.Normal('x', mu=np.array([0., 10.]), sd=np.array([2., 1.]), shape=2)
        # y = pm.HalfCauchy('y', 10.)
        # b = pm.Binomial('b', 10., 1.)

        sampler = export_to_emcee()
        start = get_start_point(sampler)

        nsteps = 1000
        for _ in tqdm(sampler.sample(start, iterations=nsteps), total=nsteps):
            pass

        from mcmc_bridge.backends import EmceeTrace
        trace = EmceeTrace(sampler)
        pm.traceplot(trace, combined=True)
        import matplotlib.pyplot as plt
        plt.show()