import numpy as np
from tqdm import tqdm
from mcmc_bridge.tests.models import correlation, linear
from mcmc_bridge.pool import InitialisedInterruptiblePool
from mcmc_bridge import EmceeTrace, export_to_emcee, get_start_point
import pytest


@pytest.mark.parametrize("test_model_function,steps,nwalker_multiple", [(linear, 1000, 12)])
@pytest.mark.parametrize("threads,use_pool", [(1, False), (4, True)])
def test_model(test_model_function, steps, nwalker_multiple, threads, use_pool):
    pymc_model, true_variables = test_model_function()
    with pymc_model:
        sampler = export_to_emcee(nwalker_multiple=nwalker_multiple, threads=threads, use_pool=use_pool)
        assert sampler.model is pymc_model
        start = get_start_point(sampler)
        assert start.shape == (sampler.k, sampler.dim)
        for _ in tqdm(sampler.sample(start, iterations=steps), total=steps):
            pass
        if use_pool:
            sampler.pool.close()
        assert sampler.chain.shape == (sampler.k, steps, sampler.dim)
        trace = EmceeTrace(sampler)
        assert set(trace.varnames) == {i.name for i in pymc_model.unobserved_RVs}
        assert len(trace) == steps
        assert trace.nchains == sampler.k
        for varname, (truth, allowed_tolerance) in true_variables.items():
            assert np.allclose(np.median(trace[varname], axis=0), truth, atol=allowed_tolerance)
