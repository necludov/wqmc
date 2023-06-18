# Wasserstein Quantum Monte Carlo (WQMC)

This repository contains the original implementation of the experiments for "Wasserstein Quantum Monte Carlo: A Novel Approach for Solving the Quantum Many-Body Schrödinger Equation".

--------------------

## Sketch of the algorithm

The solution to the stationary-time Schrödinger equation (the first eigenstate) can be found as the distribution minimizing the energy of the quantum system, i.e.
$$E_{0} = \min_q E[q],$$
where $q(x)$ is the state-space density of electrons and $E_0$ is the ground state energy.
One can define the energy-minimizing gradient flow on the space of distributions, which is a PDE defining the evolution of the density function $q_t(x)$.
However, the form of the PDE significantly depends on the metric introduced on the space of distributions.

In this paper, we are interested in a comparison of two metrics: the Fisher-Rao metric and the Wasserstein-2 metric.
Intuitively, the Fisher-Rao metric allows for probability mass _teleportation_, while Wasserstein-2 restricts the density change to mass _transportation_ (see GIFs illustrating this intuition below).

**TLDR** We demonstrate that the conventional energy minimization algorithm (Quantum Variational Monte Carlo) relies on the Fisher-Rao metric, which allows for mass teleportation and, in practice, might be unfavorable property for MCMC sampling. We propose another energy-minimizing scheme (Wasserstein Quantum Monte Carlo) designed using the Wasserstein-2 metric and restricts the density change to mass transportation. Empirically, we demonstrate better convergence properties of the proposed algorithm (see results below).

|Fisher-Rao gradient flow (QVMC)| Wasserstein-2 gradient flow (WQMC)|
|:--:|:--:|
|![](https://github.com/necludov/wqmc/blob/main/assets/fr.gif)|![](https://github.com/necludov/wqmc/blob/main/assets/w2.gif)|

## Results

Convergence of the algorithms for several molecular systems. We report the relative error in the energy estimation (top row), variance of the local energy (middle row), and gradient norm of the local energy (bottom row). Note that the local energy has to be constant in the optimum, thus, a faster vanishing gradient and variance demonstrate faster convergence to the optimum.

![](https://github.com/necludov/wqmc/blob/main/assets/plots.png)

## Running the code

1. Specify the config file describing the hyperparameters and logging settings (logging is done via W&B). Examples of configs are given [here](https://github.com/necludov/wqmc/tree/main/configs).
2. Specify the working directory path. The example below uses the working directory created by slurm (for running your job on a cluster).
3. Specify the mode: `'pretrain'` your model using Hartree-Fock approximation, then `'train'` it using the algorithm specified in configs.

```
python main.py --config configs/qvmc/li2.yml \
               --workdir $PWD/checkpoint/${SLURM_JOB_ID} \
               --mode 'train'
```
