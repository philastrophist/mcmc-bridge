from functools import reduce

import emcee
import h5py
import numpy as np
from emcee.backends import HDFBackend
from mcmc_bridge.__version__ import __version__
from pymc3.backends import NDArray
from pymc3.backends.base import MultiTrace, BaseTrace
from functools import wraps

__all__ = ['EmceeTrace']


class SWMRHDFBackend(HDFBackend):
    def open(self, mode="r"):
        if self.read_only and mode != "r":
            raise RuntimeError("The backend has been loaded in read-only "
                               "mode. Set `read_only = False` to make "
                               "changes.")
        return h5py.File(self.filename, mode, swmr=True)



class Pymc3EmceeHDF5Backend(SWMRHDFBackend):
    """A backend that stores the chain in an HDF5 file using h5py

    .. note:: You must install `h5py <http://www.h5py.org/>`_ to use this
        backend.

    Args:
        filename (str): The name of the HDF5 file where the chain will be
            saved.
        name (str; optional): The name of the group where the chain will
            be saved.
        read_only (bool; optional): If ``True``, the backend will throw a
            ``RuntimeError`` if the file is opened with write access.

    """
    def reset(self, nwalkers, ndim):
        """Clear the state of the chain and empty the backend

        Args:
            nwakers (int): The size of the ensemble
            ndim (int): The number of dimensions

        """
        with self.open("w") as f:
            g = f.create_group(self.name)
            g.attrs["version"] = __version__
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["has_blobs"] = False
            g.attrs["iteration"] = 0
            g.create_dataset("accepted", data=np.zeros(nwalkers))
            g.create_dataset("chain",
                             (0, nwalkers, ndim),
                             maxshape=(None, nwalkers, ndim),
                             dtype=np.float64)
            g.create_dataset("log_prob",
                             (0, nwalkers),
                             maxshape=(None, nwalkers),
                             dtype=np.float64)

    def has_blobs(self):
        with self.open() as f:
            return f[self.name].attrs["has_blobs"]

    def get_value(self, name, flat=False, thin=1, discard=0):
        if not self.initialized:
            raise AttributeError("You must run the sampler with "
                                 "'store == True' before accessing the "
                                 "results")
        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError("You must run the sampler with "
                                     "'store == True' before accessing the "
                                     "results")

            if name == "blobs" and not g.attrs["has_blobs"]:
                return None

            v = g[name][discard + thin - 1:self.iteration:thin]
            if flat:
                s = list(v.shape[1:])
                s[0] = np.prod(v.shape[:2])
                return v.reshape(s)
            return v

    @property
    def shape(self):
        with self.open() as f:
            g = f[self.name]
            return g.attrs["nwalkers"], g.attrs["ndim"]

    @property
    def iteration(self):
        with self.open() as f:
            return f[self.name].attrs["iteration"]

    @property
    def accepted(self):
        with self.open() as f:
            return f[self.name]["accepted"][...]

    @property
    def random_state(self):
        with self.open() as f:
            elements = [
                v
                for k, v in sorted(f[self.name].attrs.items())
                if k.startswith("random_state_")
            ]
        return elements if len(elements) else None

    def grow(self, ngrow, blobs):
        """Expand the storage space by some number of samples

        Args:
            ngrow (int): The number of steps to grow the chain.
            blobs: The current list of blobs. This is used to compute the
                dtype for the blobs array.

        """
        self._check_blobs(blobs)

        with self.open("a") as f:
            g = f[self.name]
            ntot = g.attrs["iteration"] + ngrow
            g["chain"].resize(ntot, axis=0)
            g["log_prob"].resize(ntot, axis=0)
            if blobs is not None:
                has_blobs = g.attrs["has_blobs"]
                if not has_blobs:
                    nwalkers = g.attrs["nwalkers"]
                    dt = np.dtype((blobs[0].dtype, blobs[0].shape))
                    g.create_dataset("blobs", (ntot, nwalkers),
                                     maxshape=(None, nwalkers),
                                     dtype=dt)
                else:
                    g["blobs"].resize(ntot, axis=0)
                g.attrs["has_blobs"] = True

    def save_step(self, state, accepted):
        """Save a step to the backend

        Args:
        state (State): The :class:`State` of the ensemble.
            accepted (ndarray): An array of boolean flags indicating whether
                or not the proposal for each walker was accepted.

        """
        self._check(state, accepted)

        with self.open("a") as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]

            g["chain"][iteration, :, :] = state.coords
            g["log_prob"][iteration, :] = state.log_prob
            if state.blobs is not None:
                g["blobs"][iteration, :] = state.blobs
            g["accepted"][:] += accepted

            for i, v in enumerate(state.random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1




class Indexer(object):
    def __init__(self, walker_trace):
        self.walker_trace = walker_trace
        self.emcee_backend = walker_trace.backend
        self.chain = walker_trace.chain
        self.indices = self.walker_trace.indices

    def __getitem__(self, item):
        if isinstance(item, tuple):
            varname, _slice = item
        else:
            varname = item
            _slice = slice(None)

        index = self.indices[_slice]
        try:
            with self.emcee_backend.open('r') as f:
                return f[self.emcee_backend.name]['blobs'][index, self.chain][varname]
        except AttributeError:
            return self.emcee_backend.get_blobs()[index, self.chain][varname]


class EmceeWalkerTrace(NDArray):
    def __init__(self, parent_multitrace, chain, name=None, model=None):
        self.name = name
        self.backend = parent_multitrace.backend
        self.parent_multitrace = parent_multitrace

        try:
            BaseTrace.__init__(self, name, model, None, None)
        except TypeError:
            self.model = None
            self.varnames = self.parent_multitrace.varnames
            self.fn = None
            self.vars = None

        self.chain = chain
        self._is_base_setup = True
        self.sampler_vars = None
        self._warnings = []

        self.idxs = [slice(None)]
        self.draws = self.draw_idx
        self._stats = None

    @property
    def indices(self):
        return reduce(lambda x, y: x[y], self.idxs, range(self.backend.iteration))


    @property
    def draw_idx(self):
        return len(self.indices)

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

    def _slice(self, idx):
        # Slicing directly instead of using _slice_as_ndarray to
        # support stop value in slice (which is needed by
        # iter_sample).

        # Only the first `draw_idx` value are valid because of preallocation
        idx = slice(*idx.indices(len(self)))

        sliced = EmceeWalkerTrace(self.parent_multitrace, self.chain, name=self.name, model=self.model)
        sliced.idxs.append(idx)
        return sliced

    def point(self, idx):
        return {v: self.samples[v, int(idx)] for v in self.varnames}



class EmceeTrace(MultiTrace):
    def __init__(self, backend):
        if isinstance(backend, emcee.EnsembleSampler):
            backend = backend.backend
        self.backend = backend
        traces = [EmceeWalkerTrace(self, i) for i in range(backend.shape[0])]
        super(EmceeTrace, self).__init__(traces)

    @property
    def varnames(self):
        return self.backend.get_last_sample().blobs[0].dtype.names

    @property
    def stat_names(self):
        return {}

    def get_values(self, varname, burn=0, thin=1, combine=True, chains=None, squeeze=True):
        return np.asarray(super().get_values(varname, burn, thin, combine, chains, squeeze))


# TODO: Make it so EmceeTrace only reads the variable shape/dtype once for all walkers; no need for them all to do it
# TODO: h5py with optional parallel MPI writer/