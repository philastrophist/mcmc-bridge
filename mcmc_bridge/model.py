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

    return f, supplementary_fn, [u.name for u in unobserved]


def lnpost(theta, likelihood_fn, supplementary_fn, nans='-inf'):
    assert nans == '-inf' or nans == 'raise', "nans must be either 'raise' or '-inf'"
    v = likelihood_fn(theta)
    supplementary = supplementary_fn(array2point(theta))
    if nans == '-inf' and np.isnan(v):
        return -np.inf, supplementary
    return v, supplementary


def dummy_function(*args, **kwargs):
    pass


# @contextmanager
# def EmceeMPIPool(**kwargs):
#     """
#     Context required for building pymc3 models with MPI
#     Example:
#     >>> with EmceeMPIPool(loadbalance=True):
#     ...     with pm.Model() as model:
#     ...         n = pm.Normal('n', 0, 1)
#     >>> with model:
#     ...     sampler = export_to_emcee()
#     >>> sample(sampler)
#     :param kwargs:
#     :return:
#     """
#     pool = MPIPool(**kwargs)
#     if pool.is_master():
#         # return the pool and
#         yield pool
#         with pm.modelcontext(None) as model:
#             _get_scalar_loglikelihood_functions(model)
#         print('Master: Built model. Signalling other processes')
#
#     else:
#         pool.wait()  # now wait for emcee instructions
#         sys.exit(0)




def export_to_emcee(model=None, nwalker_multiple=2, threads=1, use_pool=False, mpi_pool=None, **kwargs):
    import emcee
    model = pm.modelcontext(model)
    pool = None

    if (use_pool and (threads > 1)):
        f, sup_f, unobserved_varnames = _get_scalar_loglikelihood_functions(model)
        l = partial(lnpost, likelihood_fn=f, supplementary_fn=sup_f)
        pool = InitialisedInterruptiblePool(threads, l)
        fn = dummy_function
    elif mpi_pool is not None:
        pool = mpi_pool
        # with pool:
        #     if pool.is_master():
        #         print('master building functions')
        #         f, sup_f, unobserved_varnames = _get_scalar_loglikelihood_functions(model)
        #         l = partial(lnpost, likelihood_fn=f, supplementary_fn=sup_f)
        #         print('Built functions')
        #     else:
        #         pool.wait()
        #
        f, sup_f, unobserved_varnames = _get_scalar_loglikelihood_functions(model)
        l = partial(lnpost, likelihood_fn=f, supplementary_fn=sup_f)
        pool.worker_function = l

        fn = dummy_function
        if not pool.is_master():
            print(pool.rank, "awaiting function input")
            pool.wait()

    else:
        f, sup_f, unobserved_varnames = _get_scalar_loglikelihood_functions(model)
        fn = partial(lnpost, likelihood_fn=f, supplementary_fn=sup_f)

    dim = sum(var.dsize for var in model.vars)
    sampler = emcee.EnsembleSampler(nwalker_multiple*dim, dim, fn, pool=pool, **kwargs)
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


def trace2array(trace, model):
    return np.asarray([point2array(s, model, model.vars) for s in trace])


def start_point_from_trace(sampler, **pymc3_kwargs):
    trace = pm.sample(get_nwalkers(sampler), **pymc3_kwargs)
    return trace2array(trace, sampler.model)


def get_start_point(sampler, init='advi', n_init=500000, progressbar=True, **kwargs):
    start, _ = pm.init_nuts(init, get_nwalkers(sampler), n_init=n_init, model=sampler.model, progressbar=progressbar, **kwargs)
    return trace2array(start, sampler.model)
