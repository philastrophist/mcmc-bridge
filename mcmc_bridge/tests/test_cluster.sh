#!/bin/bash

environment="$(conda env list | grep '*' | cut -d' ' -f 1)"
echo "using current environment $environment"
qsub -v CONDA_ENV="$environment" test_cluster.qsub