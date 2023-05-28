#!/bin/bash

#SBATCH -n 20                              # Number of cores
#SBATCH --time=24:00:00                    # hours:minutes:seconds
#SBATCH --mem-per-cpu=2000
#SBATCH --tmp=4000                        # per node!!
#SBATCH --job-name=deep_mimic_$motion
#SBATCH --output=./out/deep_mimic_$motion.out
#SBATCH --error=./out/deep_mimic_$motion.err
#SBATCH --gpus=1

source scripts/setup.sh

xvfb-run -a --server-args="-screen 0 480x480x24" python -m model.main -wb -vr -c conf/pybullet_env.json -a "run_humanoid3d_$motion_args.txt"
