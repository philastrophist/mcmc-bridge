import numpy as np
from pymc3.backends import NDArray
from pymc3.backends.base import MultiTrace

__all__ = ['EmceeTrace']

class Indexer(object):
    def __init__(self, walker_trace):
        self.walker_trace = walker_trace

        self.chain = self.walker_trace.chain
        self.raw_samples = self.walker_trace.emcee_trace_obj.sampler.chain
        self.unfitted_chain = self.walker_trace.emcee_trace_obj.unfitted_chain
        self.ordering = self.walker_trace.emcee_trace_obj.sampler.ordering

    def __getitem__(self, varname):
        try:
            varmap = self.ordering[varname]
            samples = self.raw_samples[self.chain, :, varmap.slc]
            return samples.reshape(samples.shape[:1]+varmap.shp).astype(varmap.dtyp)
        except KeyError:
            return self.unfitted_chain[varname][self.walker_trace.chain]

    def items(self):
        names = [i.name for i in self.walker_trace.model.vars] + self.walker_trace.emcee_trace_obj.sampler.unobserved_varnames
        return zip(names, [self[n] for n in names])



class EmceeWalkerTrace(NDArray):
    def __init__(self, emcee_trace_obj, chain, name=None):
        super(NDArray, self).__init__(name, model=emcee_trace_obj.sampler.model)
        self.emcee_trace_obj = emcee_trace_obj
        self.chain = chain
        self.draw_idx = emcee_trace_obj.sampler.chain.shape[1]
        self.draws = self.draw_idx
        self._stats = None

    @property
    def fitted_varnames(self):
        return [i.name for i in self.emcee_trace_obj.sampler.model.vars]

    @property
    def samples(self):
        return Indexer(self)


    def setup(self, draws, chain):
        raise NotImplemented("Emcee chains are already setup")


    def record(self, point):
        raise NotImplemented("Emcee chains cannot be written to by pymc3")


    def close(self):
        raise NotImplemented("Emcee chains cannot be written to by pymc3")


def unpack_param_blobs(sampler):
    params = {n: None for n in sampler.unobserved_varnames}
    for i, iteration in enumerate(sampler.blobs[:sampler.chain.shape[1]]):
        for w, walker in enumerate(iteration):
            for p, param in zip(sampler.unobserved_varnames, walker):
                if params[p] is None:
                    params[p] = np.zeros(sampler.chain.shape[:2]+param.shape)
                params[p][w, i] = param
    return params


class EmceeTrace(MultiTrace):
    def __init__(self, sampler):
        self.sampler = sampler
        self.unfitted_chain = unpack_param_blobs(sampler)
        traces = [EmceeWalkerTrace(self, i) for i in range(sampler.chain.shape[0])]
        super(EmceeTrace, self).__init__(traces)
