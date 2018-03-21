import pymc3 as pm
import theano
from pymc3.backends.base import MultiTrace
from pymc3.backends.ndarray import NDArray


def get_scalar_loglikelihood_function(model, vars=None):
    model = pm.modelcontext(model)
    if vars is None:
        vars = model.vars
    shared = pm.make_shared_replacements(vars, model)
    [logp0], inarray0 = pm.join_nonshared_inputs([model.logpt], vars, shared)

    f = theano.function([inarray0], logp0)
    f.trust_input = True

    def wrapped(theta):
        return f(theta)

    return wrapped


def export_to_emcee(model=None, nwalker_multiple=2, **kwargs):
    import emcee
    model = pm.modelcontext(model)
    lnpost = get_scalar_loglikelihood_function(model)
    dim = model.ndim
    sampler = emcee.EnsembleSampler(nwalker_multiple*dim, dim, lnpost, **kwargs)
    sampler.model = model
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

def export_trace(sampler):
    bar = tqdm(total=sampler.chain.shape[1] * get_nwalkers(sampler))
    traces = []
    for i, chain in enumerate(sampler.chain):
        trace = NDArray(name=str(i), model=sampler.model)
        trace.setup(sampler.chain.shape[1], i)
        for apoint in chain:
            trace.record(array2point(apoint, sampler.model, sampler.model.vars))
            bar.update()
        trace.close()
        traces.append(trace)
    return MultiTrace(traces)


def get_start_point(sampler, init='advi', n_init=500000, progressbar=True, **kwargs):
    start, _ = pm.init_nuts(init=init, chains=get_nwalkers(sampler), n_init=n_init, model=sampler.model, progressbar=progressbar, **kwargs)
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
        sampler = export_to_emcee()
        start = get_start_point(sampler)

        nsteps = 1000
        from tqdm import tqdm
        for _ in tqdm(sampler.sample(start, iterations=nsteps), total=nsteps):
            pass

        trace = export_trace(sampler)
        pm.traceplot(trace, combined=False)
        import matplotlib.pyplot as plt
        plt.show()