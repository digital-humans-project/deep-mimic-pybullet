# Deep Mimic with PyBullet Environment

This repository contains a fork of PyBullet's implementation of the humanoid model used in DeepMimic, and our implementation of the RL pipeline to mimic motion from a reference motion clip on the humanoid model.

## Install

Install PyTorch using conda or pip.

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then install the package.

```bash
pip install --upgrade pip
pip install -e .
```

Please check for an installation script in `scripts/install.sh`.

## Run on Euler

Modules required to be loaded on Euler before running are:

```bash
module load gcc/8.2.0 python/3.9.9 cmake/3.25.0 freeglut/3.0.0 libxrandr/1.5.0  libxinerama/1.1.3 libxi/1.7.6  libxcursor/1.1.14 mesa/17.2.3 openmpi/4.1.4 eth_proxy
```

All sbatch scripts are in `jobs/`. Please run job scripts from the root directory of this repository since some paths coded in the scripts are relative.

## Export

Videos can be exported from a trained model using:
    
```bash
python -m model.video_exporter -c <config>
```

Sample config files can be found in `conf/`.
