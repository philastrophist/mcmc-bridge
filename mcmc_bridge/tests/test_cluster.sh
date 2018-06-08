#!/bin/bash
#PBS -N test-pymc3-emcee-bridge
#PBS -m abe
#PBS -k oe
#PBS -l nodes=2:ppn=8
#PBS -l walltime=00:30:00

echo ------------------------------------------------------
echo -n 'Job is running on node '; cat $PBS_NODEFILE
echo ------------------------------------------------------
echo PBS: qsub is running on $PBS_O_HOST
echo PBS: originating queue is $PBS_O_QUEUE
echo PBS: executing queue is $PBS_QUEUE
echo PBS: working directory is $PBS_O_WORKDIR
echo PBS: execution mode is $PBS_ENVIRONMENT
echo PBS: job identifier is $PBS_JOBID
echo PBS: job name is $PBS_JOBNAME
echo PBS: node file is $PBS_NODEFILE
echo PBS: current home directory is $PBS_O_HOME
echo PBS: PATH = $PBS_O_PATH
echo ------------------------------------------------------

source activate pymc3-uptodate
export MKL_THREADING_LAYER="GNU"

# go to test directory
cd "$(dirname "$(python -c "import mcmc_bridge as m; print(m.__file__)")")" || exit 1
cd tests || exit 1

compiledirbase="$(python -c "import theano as t; print(t.config.compiledir)")"

if [[ "$compiledirbase" = "/home/sread/.theano"* ]]; then
    echo -n "removing previous theano/compile directories for safety..."
    rm -r "$compiledirbase" || exit 1
    echo "Done"
else
    echo "Dangerous compiledirbase $compiledirbase"
    exit 1
fi

NCPUS=$(wc -l $PBS_NODEFILE | awk '{print $1}')

mpiexec python _test_cluster.py 16 8000 "$compiledirbase"

echo ------------------------------------------------------
echo Job ends