import numpy as np
import pytest
from emcee.backends import HDFBackend
from mcmc_bridge import EmceeTrace, export_to_emcee, get_start_point
from mcmc_bridge.tests.models import correlation, linear
from tqdm import tqdm


@pytest.mark.parametrize("test_model_function,steps,nwalker_multiple", [(correlation, 500, 12)])  # (linear, 100, 12)
@pytest.mark.parametrize("threads,use_pool", [(1, False)])#, (4, True)])
def test_model(test_model_function, steps, nwalker_multiple, threads, use_pool, tmpdir):
    np.random.seed(1213)
    pymc_model, true_variables = test_model_function()
    with pymc_model:
        backend = HDFBackend(str(tmpdir.mkdir("test_model").join('test.h5')))
        sampler = export_to_emcee(nwalker_multiple=nwalker_multiple, threads=threads, use_pool=use_pool, backend=backend)
        start = get_start_point(sampler)
        assert start.shape == (sampler.k, sampler.dim)
        for _ in tqdm(sampler.sample(start, iterations=steps), total=steps):
            pass
        if use_pool:
            sampler.pool.close()
        assert sampler.chain.shape == (sampler.k, steps, sampler.dim)

        trace = EmceeTrace(sampler)
        assert set(trace.varnames) == {i for i in pymc_model.named_vars.keys()}
        assert len(trace) == steps
        assert trace.nchains == sampler.k
        for varname, (truth, allowed_tolerance) in true_variables.items():
            assert np.allclose(np.median(trace[varname], axis=0), truth, atol=allowed_tolerance)