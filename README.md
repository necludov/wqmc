# Wasserstein Quantum Monte Carlo

This repository contains the original implementation of the experiments for "Wasserstein Quantum Monte Carlo: A Novel Approach for Solving the Quantum Many-Body Schrödinger Equation".

--------------------

## Sketch of the algorithm

## Results

## Running the code

1. Specify the config file describing the hyperparameters and logging settings (logging is done via W&B). Examples of configs are given [here](https://github.com/necludov/wqmc/tree/main/configs).
2. Specify the working directory path. The example below uses the working directory created by slurm (for running your job on a cluster).
3. Specify the mode: `'pretrain'` your model using Hartree-Fock approximation, then `'train'` it using the algorithm specified in configs.

```
python main.py --config configs/qvmc/li2.yml \
               --workdir $PWD/checkpoint/${SLURM_JOB_ID} \
               --mode 'train'
```
