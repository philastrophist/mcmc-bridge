#!/usr/bin/env bash

set -ex # fail on first error, print commands

while test $# -gt 0
do
    case "$1" in
        --global)
            GLOBAL=1
            ;;
        --no-setup)
            NO_SETUP=1
            ;;
    esac
    shift
done

command -v conda >/dev/null 2>&1 || {
  echo "Requires conda but it is not installed.  Run install_miniconda.sh." >&2;
  exit 1;
}

ENVNAME="testenv"
PYTHON_VERSION=${PYTHON_VERSION:-3.6} # if no python specified, use 3.6

if [ -z ${GLOBAL} ]
then
    if conda env list | grep -q ${ENVNAME}
    then
      echo "Environment ${ENVNAME} already exists, keeping up to date"
    else
      conda create -n ${ENVNAME} --yes pip python=${PYTHON_VERSION}
      source activate ${ENVNAME}
    fi
fi
conda install --yes numpy scipy mkl-service theano pygpu

pip install --upgrade pip

#  Install editable using the setup.py
pip install -r requirements.txt
pip install -e .