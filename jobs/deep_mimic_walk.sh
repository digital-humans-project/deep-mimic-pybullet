#!/bin/bash

#SBATCH -n 16                              # Number of cores
#SBATCH --time=24:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=deep_mimic_walk
#SBATCH --output=./jobs/deep_mimic_walk.out
#SBATCH --error=./jobs/deep_mimic_walk.err
#SBATCH --gpus=1

source scripts/setup.sh

xvfb-run -a --server-args="-screen 0 480x480x24" python src/python/main.py -wb -vr -c conf/pybullet_walk_env.json
