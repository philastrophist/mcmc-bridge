from _warnings import warn

import h5py

import emcee
import numpy as np
import pymc3 as pm
from mcmc_bridge.model import array2point
from pymc3.backends import NDArray
from pymc3.backends.base import MultiTrace

from pymc3.blocking import VarMap, ArrayOrdering

__all__ = ['EmceeTrace']


# TODO: clean up this mess!

class HDFCompatibleArrayOrdering(ArrayOrdering):
    def to_hdf(self, group):
        for v in self.vmap:
            vargroup = group.create_group(v.var)
            vargroup.create_dataset('dshape', data=np.asarray(v.shp))
            vargroup.attrs['dtype'] = u'{}'.format(v.dtyp)
            vargroup.create_dataset('slc', data=np.asarray([v.slc.start, v.slc.stop]))

    @classmethod
    def from_hdf(cls, group):
        ordering = cls([])
        for varname, vargroup in group.items():
            dshape = tuple(vargroup['dshape'][:].tolist())
            dtype = vargroup.attrs['dtype']
            slc = slice(*vargroup['slc'])
            vmap = VarMap(varname, slc, dshape, dtype)
            ordering.vmap.append(vmap)
            ordering.by_name[varname] = vmap

        ordering.vmap.sort(key=lambda x: x.slc.start)
        if len(ordering.vmap):
            ordering.size = slc.stop
        return ordering


class Pymc3EmceeHDF5Backend(emcee.backends.HDFBackend):
    def __init__(self, filename, name="mcmc", read_only=False):
        super(Pymc3EmceeHDF5Backend, self).__init__(filename, name, read_only)
        if self.initialized:
            with self.open('r') as f:
                group = f['model']
                self.unobserved_varnames = group.attrs['unobserved_varnames'].split('|')
                self.unobserved_varshapes = [tuple(group['unobserved_varshapes'][k]) for k in self.unobserved_varnames]
                self.ordering = HDFCompatibleArrayOrdering.from_hdf(group['ordering'])
        else:
            self.unobserved_varnames = None
            self.unobserved_varshapes = None
            self.ordering = None

    def reset(self, nwalkers, ndim):
        super().reset(nwalkers, ndim)
        model = pm.modelcontext(None)

        unobserved = list(set(model.unobserved_RVs) - set(model.vars))
        supplementary_fn = model.fastfn(unobserved)
        with model:
            unobserved_shapes = [v.shape for v in supplementary_fn(model.test_point)]
        unobserved_varnames = [i.name for i in unobserved]

        with self.open('a') as f:
            group = f.create_group('model')
            group.attrs['unobserved_varnames'] = u'|'.join(unobserved_varnames)
            unobserved_varshapes = group.create_group('unobserved_varshapes')
            for name, shape in zip(unobserved_varnames, unobserved_shapes):
                unobserved_varshapes.create_dataset(name, data=np.asarray(shape))
            ordering = group.create_group('ordering')
            HDFCompatibleArrayOrdering(model.vars).to_hdf(group=ordering)



class Indexer(object):
    def __init__(self, walker_trace):
        self.walker_trace = walker_trace
        self.emcee_backend = walker_trace.backend
        self.chain = walker_trace.chain
        self.unfitted_chain = walker_trace.parent_multitrace.unfitted_chain
        self.ordering = self.emcee_backend.ordering

    def __getitem__(self, item):
        if isinstance(item, tuple):
            varname, _slice = item
        else:
            varname = item
            _slice = slice(None)
        with self.emcee_backend.open('r') as f:
            try:
                varmap = self.ordering[varname]
                samples = f[self.emcee_backend.name]['chain'][_slice, self.chain, varmap.slc]
                return samples.reshape(samples.shape[:1]+varmap.shp).astype(varmap.dtyp)
            except KeyError:
                dslice, dshape = self.unfitted_chain[varname]
                return f[self.emcee_backend.name]['blobs'][_slice, self.chain, dslice].reshape(dshape)


class EmceeWalkerTrace(NDArray):
    def __init__(self, parent_multitrace, chain, name=None, model=None):
        self.name = name
        self.backend = parent_multitrace.backend
        self.parent_multitrace = parent_multitrace

        try:
            super(EmceeWalkerTrace, self).__init__(name, model, None, None)
        except TypeError:
            self.model = None
            self.varnames = self.backend.unobserved_varnames + list(self.backend.ordering.by_name.keys())
            self.fn = None

        self.chain = chain
        self._is_base_setup = True
        self.sampler_vars = None
        self._warnings = []

        self.chain = chain
        self.draw_idx = self.backend.iteration
        self.draws = self.draw_idx
        self._stats = None


    @property
    def samples(self):
        return Indexer(self)  # pretend to be a point dictionary


    def setup(self, draws, chain, sampler_vars=None):
        raise NotImplemented("Emcee chains are already setup")


    def record(self, point, sampler_stats=None):
        raise NotImplemented("Emcee chains cannot be written to by pymc3")


    def close(self):
        raise NotImplemented("Emcee chains cannot be written to by pymc3")

    def get_values(self, varname, burn=0, thin=1):
        return self.samples[varname, burn::thin]



def unpack_param_blobs(backend):
    params = {}
    previous_size = 0
    for varname, varshape in zip(backend.unobserved_varnames, backend.unobserved_varshapes):
        size = np.product(varshape, dtype=int)
        with backend.open('r') as f:
            if len(backend.unobserved_varnames) == 1:
                sl = Ellipsis
            else:
                sl = slice(previous_size, previous_size + size)
            params[varname] = (sl, f[backend.name]['blobs'].shape[:1]+varshape)
        previous_size += size
    return params


class EmceeTrace(MultiTrace):
    def __init__(self, backend):
        if isinstance(backend, emcee.EnsembleSampler):
            backend = backend.backend
        self.backend = backend
        self.unfitted_chain = unpack_param_blobs(backend)
        traces = [EmceeWalkerTrace(self, i) for i in range(backend.shape[0])]
        super(EmceeTrace, self).__init__(traces)

    @property
    def varnames(self):
        return self.backend.unobserved_varnames + list(self.backend.ordering.by_name.keys())

    @property
    def stat_names(self):
        return {}