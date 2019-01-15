#!/usr/bin/env bash

set -xe

MYDIR="$(cd "$(dirname "$0")"; pwd -P)"

source ci_support/setup_conda.sh

conda install conda-build=3 -c defaults --yes

export CPU_COUNT=2

conda info -a
conda list
conda build purge-all

sed -e "s/CONFIG/${CONFIG}/" recipes/eman/conda_build_config.yaml.templ > conda_build_config.yaml

conda build recipes/eman -c cryoem -c defaults -c conda-forge -m conda_build_config.yaml
