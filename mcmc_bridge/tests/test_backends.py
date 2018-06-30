import os

from mcmc_bridge import get_start_point, export_to_emcee
from mcmc_bridge.backends import HDFCompatibleArrayOrdering, Pymc3EmceeHDF5Backend, EmceeTrace
from emcee.backends import TempHDFBackend
import pymc3 as pm
import numpy as np
import pytest
import tempfile

from mcmc_bridge.model import start_point_from_trace


def test_model():
    with pm.Model() as model:
        pm.Normal('normal', mu=np.ones((2,3)), sd=np.ones((2, 3)), shape=(2,3))
        pm.HalfCauchy('halfcauchy', beta=1)
        pm.Binomial('binomial', n=2, p=0.5)
    return model

def test_array_ordering():
    model = test_model()

    ordering = pm.ArrayOrdering(model.vars)
    hdf_ordering = HDFCompatibleArrayOrdering(model.vars)

    with TempHDFBackend() as backend:
        with backend.open('a') as f:
            group = f.create_group('test')
            hdf_ordering.to_hdf(group)

            out_hdf_ordering = HDFCompatibleArrayOrdering.from_hdf(group)

        assert len(ordering.vmap) == len(out_hdf_ordering.vmap)

        for in_vmap, out_vmap in zip(ordering.vmap, out_hdf_ordering.vmap):
            for attr in ['var', 'slc', 'shp', 'dtyp']:
                assert getattr(in_vmap, attr) == getattr(out_vmap, attr)


@pytest.fixture("module")
def temp_filename():
    fname = next(tempfile._get_candidate_names())
    yield fname
    os.remove(fname)


def test_pymc3_emcee_hdf_backend_shapes_names_without_model_context(temp_filename):
    model = test_model()
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
        assert str(trace[v.name].dtype) == v.dtype

