import warnings
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

    def get_value(self, name, flat=False, thin=1, discard=0, chain_slc=None, step_slc=None, squeeze=True):
        if not self.initialized:
            raise AttributeError("You must run the sampler with "
                                 "'store == True' before accessing the "
                                 "results")
        if chain_slc is None:
            chain_slc = slice(None)
        if step_slc is None:
            step_slc = slice(discard+thin-1, self.iteration, thin)

        with self.open() as f:
            g = f[self.name]
            iteration = g.attrs["iteration"]
            if iteration <= 0:
                raise AttributeError("You must run the sampler with "
                                     "'store == True' before accessing the "
                                     "results")

            if name == "blobs":
                if not g.attrs["has_blobs"]:
                    return None
                dtypes = [(name, f[name][k].dtype, f[name].shape[2:]) for k in g[name].keys()]
                data = [g[name][i][step_slc, chain_slc] for i in g[name].keys()]
                v = np.array(data, dtype=dtypes)
            else:
                v = g[name][step_slc, chain_slc]
            if flat:
                v = np.reshape(v, (-1,) + v.shape[2:])
            if squeeze:
                if v.shape[0] == 1:
                    return v[0]
            return v

    def get_blobs(self, name=None, **kwargs):
        """Get the chain of blobs for each sample in the chain

        Args:
            flat (Optional[bool]): Flatten the chain across the ensemble.
                (default: ``False``)
            thin (Optional[int]): Take only every ``thin`` steps from the
                chain. (default: ``1``)
            discard (Optional[int]): Discard the first ``discard`` steps in
                the chain as burn-in. (default: ``0``)

        Returns:
            array[..., nwalkers]: The chain of blobs.

        """
        if name is None:
            name = ""
        return self.get_value("blobs/"+name, **kwargs)

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
                dt = np.dtype((blobs[0].dtype, blobs[0].shape))
                if not has_blobs:
                    g.create_group("blobs")
                    nwalkers = g.attrs["nwalkers"]
                    for name in dt.names:
                        dtype, shape = dt[name].subdtype
                        g.create_dataset(name, shape=(ntot, nwalkers) + shape, dtype=dtype, maxshape=(None, nwalkers) + shape)
                else:
                    for name in dt.names:
                        g["blobs"][name].resize(ntot, axis=0)
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
                for name in state.blobs.names:
                    g["blobs"][name][iteration, ...] = state.blobs[name]
            g["accepted"][:] += accepted

            for i, v in enumerate(state.random_state):
                g.attrs["random_state_{0}".format(i)] = v

            g.attrs["iteration"] = iteration + 1


class EmceeTrace(MultiTrace):
    def __init__(self, backend):
        super(EmceeTrace, self).__init__([])
        if isinstance(backend, emcee.EnsembleSampler):
            backend = backend.backend
        assert isinstance(backend, Pymc3EmceeHDF5Backend)
        self.backend = backend

    @property
    def nchains(self):
        return self.backend.nwalkers

    @property
    def chains(self):
        return range(len(self.nchains))

    def __len__(self):
        return self.backend.iteration

    @property
    def stat_names(self):
        return {}

    def add_values(self, vals):
        raise AttributeError("Cannot add to Emcee")

    def get_sampler_stats(self, varname, burn=0, thin=1, combine=True,
                          chains=None, squeeze=True):
        raise AttributeError("No stats")


    @property
    def varnames(self):
        with self.backend.open('r'):
            return list(self.backend[self.backend.name].keys())

    def get_values(self, varname, burn=0, thin=1, combine=True, chains=None, squeeze=True):
        return self.backend.get_blobs(varname, discard=burn, thin=thin, flat=combine, chains=chains, squeeze=squeeze)

    def point(self, idx, chain=None):
        array = self.backend.get_blobs(step_slc=idx, chain_slc=chain)
        return {name: array[name] for name in array.dtype.names}

    def points(self, chains=None):
        raise AttributeError("Cannot make an iterator, that defeats the point of HDF5")

    def _slice(self, slice):
        raise AttributeError("Cannot perform slice abstractly, use `get_values`")




# TODO: Make it so EmceeTrace only reads the variable shape/dtype once for all walkers; no need for them all to do it
# TODO: h5py with optional parallel MPI writer/