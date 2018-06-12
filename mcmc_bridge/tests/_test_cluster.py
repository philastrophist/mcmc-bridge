import os
import sys

import mpi4py.MPI as MPI
import numpy as np

rank = MPI.COMM_WORLD.Get_rank()
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('nwalker_multiple', type=int)
parser.add_argument('steps', type=int)
parser.add_argument('compiledirbase', type=str)

args = parser.parse_args()


d = 'compiledir={}'.format(args.compiledirbase+'/'+str(rank))

os.environ['THEANO_FLAGS'] = d
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import theano
print(theano.config.compiledir)

from tqdm import tqdm
from mcmc_bridge.tests.models import linear
from mcmc_bridge import export_to_emcee, get_start_point, EmceeTrace
from mcmc_bridge.pool import InitialisedMPIPool
import time

pymc_model, true_variables = linear()
print(rank, "compiled model")

pool = InitialisedMPIPool()

with pymc_model:
    with pool.kill_workers_on_close():
        sampler = export_to_emcee(nwalker_multiple=args.nwalker_multiple, mpi_pool=pool)
        if pool.is_master():
            start = get_start_point(sampler)
            start_time = time.time()
            for _ in tqdm(sampler.sample(start, iterations=args.steps), total=args.steps):
                pass

    print("MPI sampling complete in {:.1f}s".format(time.time() - start_time))
    trace = EmceeTrace(sampler)

    sampler = export_to_emcee(nwalker_multiple=args.nwalker_multiple)
    start_time = time.time()
    for _ in tqdm(sampler.sample(start, iterations=args.steps), total=args.steps):
        pass

    print("Serial sampling complete in {:.1f}s".format(time.time() - start_time))