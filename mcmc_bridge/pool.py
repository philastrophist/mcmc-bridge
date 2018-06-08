import sys
from contextlib import contextmanager

from emcee.interruptible_pool import InterruptiblePool
from schwimmbad.mpi import MPIPool, _dummy_callback, log, _VERBOSE
try:
    import mpi4py.MPI as MPI
except ImportError:
    MPI = None


def initialise_global_fn(fn):
    # print('initialising', fn)
    global global_fn
    global_fn = fn
    # print('finished')


def call_global_fn(*args, **kwargs):
    return global_fn(*args, **kwargs)


class InitialisedInterruptiblePool(InterruptiblePool):
    def __init__(self, processes, function, **kwargs):
        super().__init__(processes, initialise_global_fn, [function], **kwargs)

    def map(self, func, iterable, chunksize=None):
        return super().map(call_global_fn, iterable, chunksize)


class InitialisedMPIPool(MPIPool):
    """A processing pool that distributes tasks using MPI.

    With this pool class, the master process distributes tasks to worker
    processes using an MPI communicator. This pool therefore supports parallel
    processing on large compute clusters and in environments with multiple
    nodes or computers that each have many processor cores.

    This implementation is inspired by @juliohm in `this module
    <https://github.com/juliohm/HUM/blob/master/pyhum/utils.py#L24>`_

    Parameters
    ----------
    comm : :class:`mpi4py.MPI.Comm`, optional
        An MPI communicator to distribute tasks with. If ``None``, this uses
        ``MPI.COMM_WORLD`` by default.
    """

    def __init__(self, comm=None):
        super().__init__(comm)
        self.worker_function = None

    @contextmanager
    def kill_workers_on_close(self, exitcode=0):
        yield
        self.close()
        if not self.is_master():
            sys.exit(exitcode)

    def wait_and_exit(self):
        if not self.is_master():
            self.wait()
            print(self.rank, "shutdown")
            sys.exit(0)


    def wait(self, callback=None):
        """Tell the workers to wait and listen for the master process. This is
        called automatically when using :meth:`MPIPool.map` and doesn't need to
        be called by the user.
        """
        if self.is_master():
            return

        worker = self.comm.rank
        status = MPI.Status()
        while True:
            log.log(_VERBOSE, "Worker {0} waiting for task".format(worker))

            task_args = self.comm.recv(source=self.master, tag=MPI.ANY_TAG,
                                       status=status)

            if task_args is None:
                log.log(_VERBOSE, "Worker {0} told to quit work".format(worker))
                break

            log.log(_VERBOSE, "Worker {0} got task {1} with tag {2}"
                    .format(worker, task_args, status.tag))

            result = self.worker_function(task_args)

            log.log(_VERBOSE, "Worker {0} sending answer {1} with tag {2}"
                    .format(worker, result, status.tag))

            self.comm.ssend(result, self.master, status.tag)

        if callback is not None:
            callback()


    def map(self, worker, tasks, callback=None):
        """Evaluate a function or callable on each task in parallel using MPI.

        The callable, ``worker``, is called on each element of the ``tasks``
        iterable. The results are returned in the expected order (symmetric with
        ``tasks``).

        Parameters
        ----------
        worker : callable
            A function or callable object that is executed on each element of
            the specified ``tasks`` iterable. This object must be picklable
            (i.e. it can't be a function scoped within a function or a
            ``lambda`` function). This should accept a single positional
            argument and return a single object.
        tasks : iterable
            A list or iterable of tasks. Each task can be itself an iterable
            (e.g., tuple) of values or data to pass in to the worker function.
        callback : callable, optional
            An optional callback function (or callable) that is called with the
            result from each worker run and is executed on the master process.
            This is useful for, e.g., saving results to a file, since the
            callback is only called on the master thread.

        Returns
        -------
        results : list
            A list of results from the output of each ``worker()`` call.
        """

        # If not the master just wait for instructions.
        if not self.is_master():
            self.wait()
            return

        if callback is None:
            callback = _dummy_callback

        workerset = self.workers.copy()
        tasklist = [(tid, arg) for tid, arg in enumerate(tasks)]
        resultlist = [None] * len(tasklist)
        pending = len(tasklist)

        while pending:
            if workerset and tasklist:
                worker = workerset.pop()
                taskid, task_args = tasklist.pop()
                log.log(_VERBOSE, "Sent task %s to worker %s with tag %s",
                        task_args[1], worker, taskid)
                self.comm.send(task_args, dest=worker, tag=taskid)

            if tasklist:
                flag = self.comm.Iprobe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
                if not flag:
                    continue
            else:
                self.comm.Probe(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)

            status = MPI.Status()
            result = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG,
                                    status=status)
            worker = status.source
            taskid = status.tag
            log.log(_VERBOSE, "Master received from worker %s with tag %s",
                    worker, taskid)

            callback(result)

            workerset.add(worker)
            resultlist[taskid] = result
            pending -= 1

        return resultlist


    def close(self):
        """ Tell all the workers to quit."""
        if self.is_worker():
            return

        for worker in self.workers:
            self.comm.send(None, worker, 0)
