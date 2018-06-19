import sys
from functools import partial

import numpy as np
import pymc3 as pm
import theano
import theano.tensor as tt
from emcee.interruptible_pool import InterruptiblePool
from emcee.mpi_pool import MPIPool
from mcmc_bridge.pool import InitialisedInterruptiblePool
from pymc3.backends.base import MultiTrace
from pymc3.backends.ndarray import NDArray
from tqdm import tqdm
from contextlib import contextmanager

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


def _get_scalar_loglikelihood_functions(model, vars=None):
    model = pm.modelcontext(model)
    if vars is None:
        vars = model.vars
    shared = pm.make_shared_replacements(vars, model)
    [logp0], inarray0 = pm.join_nonshared_inputs([model.logpt], vars, shared)

    f = theano.function([inarray0], logp0)
    f.trust_input = True

    unobserved = list(set(model.unobserved_RVs) - set(model.vars))
    supplementary_fn = model.fastfn(unobserved)
    with model:
        shapes = [v.shape for v in supplementary_fn(model.test_point)]

    return f, supplementary_fn, [u.name for u in unobserved], shapes


def lnpost(theta, likelihood_fn, supplementary_fn, nans='-inf'):
    assert nans == '-inf' or nans == 'raise', "nans must be either 'raise' or '-inf'"
    v = likelihood_fn(theta)
    supplementary = supplementary_fn(array2point(theta))
    supplementary = np.concatenate([i.ravel() for i in supplementary])
    if nans == '-inf' and np.isnan(v):
        return -np.inf,  supplementary
    return v, supplementary


def dummy_function(*args, **kwargs):
    pass


def backwards_compatible(sampler):
    if hasattr(sampler, 'k'):
        sampler.nwalkers = sampler.k
    else:
        sampler.k = sampler.nwalkers

    if hasattr(sampler, 'dim'):
        sampler.ndim = sampler.dim
    else:
        sampler.dim = sampler.ndim


def export_to_emcee(model=None, nwalker_multiple=2, threads=1, use_pool=False, mpi_pool=None, **kwargs):
    import emcee
    model = pm.modelcontext(model)
    pool = None

    if (use_pool and (threads > 1)):
        f, sup_f, unobserved_varnames, unobserved_shapes = _get_scalar_loglikelihood_functions(model)
        l = partial(lnpost, likelihood_fn=f, supplementary_fn=sup_f)
        pool = InitialisedInterruptiblePool(threads, l)
        fn = dummy_function
    elif mpi_pool is not None:
        pool = mpi_pool
        f, sup_f, unobserved_varnames, unobserved_shapes = _get_scalar_loglikelihood_functions(model)
        l = partial(lnpost, likelihood_fn=f, supplementary_fn=sup_f)
        pool.worker_function = l
        if not pool.is_master():
            print(pool.rank, "awaiting function input")
            pool.wait()
        fn = dummy_function
    else:
        f, sup_f, unobserved_varnames, unobserved_shapes = _get_scalar_loglikelihood_functions(model)
        fn = partial(lnpost, likelihood_fn=f, supplementary_fn=sup_f)

    dim = sum(var.dsize for var in model.vars)
    sampler = emcee.EnsembleSampler(nwalker_multiple*dim, dim, fn, pool=pool, **kwargs)
    backwards_compatible(sampler)
    sampler.model = model
    sampler.unobserved_varnames = unobserved_varnames
    sampler.ordering = pm.ArrayOrdering(model.vars)
    sampler.unobserved_varshapes = unobserved_shapes
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


def get_ndim(sampler):
    try:
        return sampler.dim
    except AttributeError:
        return sampler.ndim


def trace2array(trace, model):
    return np.asarray([point2array(s, model, model.vars) for s in trace])


def start_point_from_trace(sampler, **pymc3_kwargs):
    trace = pm.sample(get_nwalkers(sampler), **pymc3_kwargs)
    return trace2array(trace, sampler.model)


def metropolis_start_points(sampler, sd, n=100, scaling=0.001, seed=None, model=None):
    step = pm.Metropolis(proposal_dist=pm.NormalProposal, scaling=scaling)
    for s in step.methods:
        s.proposal_dist.s = sd
    trace = pm.sample(get_nwalkers(sampler)*n, step=step, tune=False, cores=1, chains=1, random_seed=seed)
    array = trace2array(trace, pm.modelcontext(model))
    good = np.isfinite(trace.accept).all(axis=1)
    np.random.seed(seed)
    chosen = np.random.choice(len(good), size=get_nwalkers(sampler))
    array = array[good][chosen]
    assert array.shape[0] == get_nwalkers(sampler), "Metropolis didn't sample enough good values {}/{}".format(good.sum(), len(trace))
    return array


def get_start_point(sampler, init='advi', n_init=500000, progressbar=True, **kwargs):
    start, _ = pm.init_nuts(init, get_nwalkers(sampler), n_init=n_init, model=sampler.model, progressbar=progressbar, **kwargs)
    return trace2array(start, sampler.model)


