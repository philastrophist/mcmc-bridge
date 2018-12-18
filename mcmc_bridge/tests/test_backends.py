import os

from mcmc_bridge import get_start_point, export_to_emcee
from mcmc_bridge.backends import EmceeTrace, Pymc3EmceeHDF5Backend
import pymc3 as pm
import numpy as np
import pytest
import tempfile

from mcmc_bridge.model import start_point_from_trace


def _test_model():
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


def test_pymc3_emcee_hdf_backend_shapes_names_without_model_context(temp_filename):
    model = _test_model()
    iterations = 10
    with model:
        backend = Pymc3EmceeHDF5Backend(temp_filename)
        sampler = export_to_emcee(nwalker_multiple=2, backend=backend)
        start = start_point_from_trace(sampler.nwalkers)
        sampler.run_mcmc(start, iterations)

    backend = Pymc3EmceeHDF5Backend(temp_filename)  # reload the backend after everything is closed
    trace = EmceeTrace(backend)

    for k, v in model.named_vars.items():
        with model:
            fn = model.fastfn(v)
            shape = fn(model.test_point).shape
        assert trace[v.name].shape == (iterations * sampler.nwalkers, ) + shape
        assert trace.get_values(v.name, burn=1, thin=2, combine=True).shape == ((iterations-1) // 2 * sampler.nwalkers, ) + shape
        assert str(trace[v.name].dtype) == v.dtype

