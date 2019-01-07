import os

from mcmc_bridge import get_start_point, export_to_emcee
from mcmc_bridge.backends import EmceeTrace, Pymc3EmceeHDF5Backend
import pymc3 as pm
import numpy as np
import pytest
import tempfile

from mcmc_bridge.model import start_point_from_trace, trace2array, array2point, point2array, \
    _get_scalar_loglikelihood_functions


@pytest.fixture("module")
def pymc3_model():
    with pm.Model() as model:
        pm.Normal('normal', mu=np.ones((2,3)), sd=np.ones((2, 3)), shape=(2,3))
        pm.HalfCauchy('halfcauchy', beta=np.ones((3, 2)), shape=(3, 2))
        pm.Binomial('binomial', n=2, p=0.5)
        pm.Dirichlet('dirchlet', np.ones(6), shape=6)
    return model


@pytest.fixture("module")
def temp_filename():
    fname = next(tempfile._get_candidate_names())
    yield fname
    os.remove(fname)


def test_point2array_conversion(pymc3_model):
    with pymc3_model:
        point = pymc3_model.test_point
        array = point2array(point)
        new_point = array2point(array)
    for k, v in point.items():
        np.testing.assert_allclose(new_point[k], v)


def test_trace2array_conversion(pymc3_model):
    with pymc3_model:
        trace = pm.sample(1, tune=0, cores=1, njobs=1, chains=1)
        point = trace.point(0)
        point_array = point2array(point)
        trace_array = trace2array(trace, pymc3_model)
        np.testing.assert_allclose(point_array, trace_array[0])


def test_logp_output(pymc3_model):
    with pymc3_model:
        sampler = export_to_emcee(pymc3_model)
        output = sampler.log_prob_fn(point2array(pymc3_model.test_point))
        assert isinstance(output[0], float)


def test_named_vars_output(pymc3_model):
    with pymc3_model:
        sampler = export_to_emcee(pymc3_model)
        output = sampler.log_prob_fn(point2array(pymc3_model.test_point))[1:]
        for blob_dtype, variable in zip(sampler.blobs_dtype, output):
            assert np.shape(variable) == blob_dtype[2]
            assert np.asarray(variable).dtype == blob_dtype[1]


@pytest.mark.parametrize("backend", ['emcee', 'pymc3'])
def test_emcee_coord_equal_to_stored_chain(pymc3_model, backend, temp_filename):
    with pymc3_model:
        if backend == 'emcee':
            sampler = export_to_emcee()
        elif backend == 'pymc3':
            backend = Pymc3EmceeHDF5Backend(temp_filename)
            sampler = export_to_emcee(backend=backend)

        initial = np.asarray([point2array(pymc3_model.test_point) for i in range(sampler.nwalkers)])
        for state in sampler.sample(initial, iterations=3):
            np.testing.assert_allclose(sampler.get_last_sample().coords, state.coords)


@pytest.mark.parametrize("backend", ['emcee', 'pymc3'])
def test_emcee_blob_equal_to_stored_blob(pymc3_model, backend, temp_filename):
    with pymc3_model:
        if backend == 'emcee':
            sampler = export_to_emcee()
        elif backend == 'pymc3':
            backend = Pymc3EmceeHDF5Backend(temp_filename)
            sampler = export_to_emcee(backend=backend)

        initial = np.asarray([point2array(pymc3_model.test_point) for i in range(sampler.nwalkers)])
        for state in sampler.sample(initial, iterations=3):
            assert state.blobs.shape == (sampler.nwalkers, )
            last_sample = sampler.get_last_sample()
            sampler_blobs = last_sample.blobs
            for name in state.blobs.dtype.names:
                np.testing.assert_allclose(sampler_blobs[name], state.blobs[name])


def test_Pymc3EmceeHDF5Backend_get_values_values(temp_filename, pymc3_model):
    iterations = 5
    with pymc3_model:
        backend = Pymc3EmceeHDF5Backend(temp_filename)
        sampler = export_to_emcee(backend=backend)
        initial = start_point_from_trace(sampler.nwalkers, cores=1, chains=1)

        for i, state in enumerate(sampler.sample(initial, iterations=iterations)):
            backend = Pymc3EmceeHDF5Backend(temp_filename)  # reload the backend after everything is closed
            trace = EmceeTrace(backend)

            for k, v in pymc3_model.named_vars.items():
                assert np.allclose(trace.get_values(v.name, burn=i), state.blobs[v.name])


def test_Pymc3EmceeHDF5Backend_get_values_shapes_dtypes(temp_filename, pymc3_model):
    iterations = 5
    with pymc3_model:
        backend = Pymc3EmceeHDF5Backend(temp_filename)
        sampler = export_to_emcee(backend=backend)
        initial = start_point_from_trace(sampler.nwalkers, cores=1, chains=1)
        sampler.run_mcmc(initial, iterations)

        backend = Pymc3EmceeHDF5Backend(temp_filename)  # reload the backend after everything is closed
        trace = EmceeTrace(backend)

        for k, v in pymc3_model.named_vars.items():
            with pymc3_model:
                fn = pymc3_model.fastfn(v)
                shape = fn(pymc3_model.test_point).shape
            assert trace[v.name].shape == (iterations * sampler.nwalkers, ) + shape
            assert trace.get_values(v.name, burn=1, thin=2, combine=True).shape == ((iterations-1) // 2 * sampler.nwalkers, ) + shape
            assert trace.get_values(v.name, burn=1, thin=2, combine=False).shape == ((iterations-1) // 2, sampler.nwalkers, ) + shape
            assert str(trace[v.name].dtype) == v.dtype




@pytest.mark.parametrize("chains", [0, 1, None, slice(3)])
def test_Pymc3EmceeHDF5Backend_point(temp_filename, pymc3_model, chains):
    iterations = 5
    with pymc3_model:
        backend = Pymc3EmceeHDF5Backend(temp_filename)
        sampler = export_to_emcee(backend=backend)
        initial = start_point_from_trace(sampler.nwalkers, cores=1, chains=1)
        sampler.run_mcmc(initial, iterations)

    backend = Pymc3EmceeHDF5Backend(temp_filename)  # reload the backend after everything is closed
    trace = EmceeTrace(backend)

    if isinstance(chains, (int)):
        nwalkers = tuple()
    elif chains is None:
        nwalkers = (sampler.nwalkers, )
    elif isinstance(chains, slice):
        nwalkers = (len(range(sampler.nwalkers)[chains]), )

    for i in range(len(trace)):
        point = trace.point(i, chain=chains)
        for k, v in pymc3_model.named_vars.items():
            with pymc3_model:
                fn = pymc3_model.fastfn(v)
                shape = fn(pymc3_model.test_point).shape

            assert point[v.name].shape == nwalkers + shape
            assert point[v.name].dtype == v.dtype