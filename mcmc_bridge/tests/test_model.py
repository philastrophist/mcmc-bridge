import numpy as np

from mcmc_bridge.tests.models import correlation, linear
from mcmc_bridge import EmceeTrace, export_to_emcee, get_start_point
import pytest


@pytest.mark.parametrize("test_model_function,steps,nwalker_multiple", [(linear, 1000, 4), (correlation, 2000, 10)])
@pytest.mark.parametrize("threads,pool", [(1, False)])#, (4, False), (4, True)])
def test_model(test_model_function, steps, nwalker_multiple, threads, pool):
    pymc_model, true_variables = test_model_function()
    if pool:
        from emcee.interruptible_pool import InterruptiblePool
        pool = InterruptiblePool(threads)
    else:
        pool = None
    with pymc_model:
        sampler = export_to_emcee(nwalker_multiple=nwalker_multiple, threads=threads, pool=pool)
        assert sampler.model is pymc_model
        start = get_start_point(sampler)
        assert start.shape == (sampler.k, sampler.dim)
        sampler.run_mcmc(start, steps)
        assert sampler.chain.shape == (sampler.k, steps, sampler.dim)
        trace = EmceeTrace(sampler)
        assert set(trace.varnames) == {i.name for i in pymc_model.unobserved_RVs}
        assert len(trace) == steps
        assert trace.nchains == sampler.k
        for varname, (truth, allowed_tolerance) in true_variables.items():
            assert np.allclose(np.median(trace[varname], axis=0), truth, atol=allowed_tolerance)

