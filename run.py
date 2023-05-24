import os
import json
from time import time

import numpy as np
import jax
import jax.numpy as jnp
from jax import random

import wandb
import kfac_jax
import flax.jax_utils as flax_utils
from flax.training import checkpoints
from tqdm import trange

from systems_catalog import system_catalog
import train_utils as tutils
import phys_utils as putils
from models import utils as mutils
import mcmc


def preshape(x):
  num_dev = jax.local_device_count()
  return x.reshape((num_dev, x.shape[0]//num_dev) + x.shape[1:])

def preshape_inv(x):
  num_dev = jax.local_device_count()
  return x.reshape((x.shape[1]*num_dev,) + x.shape[2:])


def pretrain(config, workdir):
  molecule = system_catalog[config.dim][config.system]
  n_electrons = molecule.n_up_electrons + molecule.n_down_electrons
  nspins = jnp.array([molecule.n_up_electrons, molecule.n_down_electrons])
  config.model.dim = config.dim*n_electrons
  key = random.PRNGKey(config.seed)

  # init
  key, *init_keys = random.split(key, 4)
  logp0, orbitals0 = putils.get_init_psi(config)
  model, initial_params = mutils.init_model(init_keys[0], config)
  loss_fn = tutils.get_pretrain_loss(config, model)
  init_samples = mcmc.get_init_samples(init_keys[1], config)
  optimizer, opt_state = tutils.get_optimizer(config.pretrain, loss_fn, init_keys[2], initial_params, init_samples)
  
  state = tutils.QMCState(step=1, opt_state=opt_state,
    model_params=initial_params, key=key,
    wandbid=np.random.randint(int(1e7),int(1e8)),
    sigma=config.mcmc.rwmh_sigma, samples=init_samples
  )

  checkpoint_dir = os.path.join(workdir, "checkpoints")
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, prefix='premodel_')
  init_step = int(state.step)
  key = state.key

  kernel = mcmc.get_RWMH_kernel(config, lambda _p, _x: logp0(_x))
  if init_step == 1:
    sigma = config.mcmc.rwmh_sigma
    key, init_key = random.split(key)
    samples = mcmc.get_init_samples(init_key, config)
    ar_plot = np.zeros(config.mcmc.n_init_steps)
    for iter in trange(config.mcmc.n_init_steps):
      key, mcmc_key = random.split(key)
      samples, AR = kernel(mcmc_key, None, samples, sigma)
      sigma = mcmc.update_sigma(AR, sigma)
    state = state.replace(sigma=sigma, samples=samples)

  if jax.process_index() == 0:
    wandb.init(id=str(state.wandbid),
               project=config.system,
               resume="allow",
               config=config)
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = str(state.wandbid)

  state = state.replace(model_params=flax_utils.replicate(state.model_params))
  key = jax.random.fold_in(key, jax.process_index())
  step_fn = tutils.get_train_step(config.pretrain, loss_fn, optimizer)
  for step in range(init_step, config.pretrain.n_iter + 1):
    state = state.replace(step=step)

    # mcmc step
    key, mcmc_key = random.split(key)
    samples = state.samples
    samples, AR = kernel(mcmc_key, None, samples, sigma)
    sigma = mcmc.update_sigma(AR, sigma)
    state = state.replace(sigma=sigma, samples=samples)
    
    batch = (preshape(samples), preshape(orbitals0(samples)))
    key, *next_key = jax.random.split(key, num=jax.local_device_count() + 1)
    next_key = jnp.asarray(next_key)

    state, loss, aux = step_fn(next_key, state, batch)
    if jax.process_index() == 0:
      wandb.log({'pretrain loss': loss.mean().item(), 'AR': AR.item(), 'sigma': sigma}, step=step)
    if (step % config.pretrain.save_every == 0) and (jax.process_index() == 0):
      saved_state = state.replace(model_params=flax_utils.unreplicate(state.model_params),
        opt_state=flax_utils.unreplicate(state.opt_state), key=key)
      checkpoints.save_checkpoint(checkpoint_dir, saved_state, 
                                  step=step//config.pretrain.save_every, 
                                  prefix='premodel_')

  def model_logdensity(params, x):
    return 2*model.apply(params, x, False)[1][:, None]
  kernel = mcmc.get_RWMH_kernel(config, model_logdensity)
  kernel = jax.pmap(kernel, in_axes=(0,0,0,None))
  samples = preshape(samples)
  # init samples
  print('Generating samples from pretrained model...')
  for iter in range(config.mcmc.n_init_steps):
    key, *mcmc_key = random.split(key, num=jax.local_device_count() + 1)
    mcmc_key = jnp.asarray(mcmc_key)
    samples, AR = kernel(mcmc_key, state.model_params, samples, sigma)
    sigma = mcmc.update_sigma(AR, sigma)
  samples = preshape_inv(samples)
  
  if jax.process_index() == 0:
    saved_state = state.replace(model_params=flax_utils.unreplicate(state.model_params),
      opt_state=flax_utils.unreplicate(state.opt_state), key=key, samples=samples, sigma=sigma
    )
    checkpoints.save_checkpoint(checkpoint_dir, saved_state, 
      step=config.pretrain.n_iter//config.pretrain.save_every + 1, 
      prefix='premodel_', keep=2
    )

def train(config, workdir):
  molecule = system_catalog[config.dim][config.system]
  n_electrons = molecule.n_up_electrons + molecule.n_down_electrons
  nspins = jnp.array([molecule.n_up_electrons, molecule.n_down_electrons])
  config.model.dim = config.dim*n_electrons
  key = random.PRNGKey(config.seed)
  V = putils.get_potential(config)

  # load pretrain
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  state = checkpoints.restore_checkpoint(checkpoint_dir, None, prefix='premodel_')
  params = state['model_params']
  samples = preshape(state['samples'])

  # init
  key, init_key = random.split(key)
  model, _ = mutils.init_model(init_key, config)
  loss_fn = tutils.get_loss(config, model, V)
  optimizer, opt_state = tutils.get_optimizer(config.train, loss_fn, key, params, samples)
  state = tutils.QMCState(step=1, opt_state=opt_state, 
    model_params=flax_utils.replicate(params),
    key=key, wandbid=np.random.randint(int(1e7),int(1e8)), 
    samples=samples, sigma=state['sigma'])
  
  # load preemption
  state = checkpoints.restore_checkpoint(checkpoint_dir, state, prefix='model_')
  init_step = int(state.step)
  key = state.key
  optimizer, _ = tutils.get_optimizer(config.train, loss_fn, key, flax_utils.unreplicate(state.model_params), state.samples)

  if jax.process_index() == 0:
    wandb.init(id=str(state.wandbid),
               project=config.system,
               resume="allow",
               config=config)
    os.environ["WANDB_RESUME"] = "allow"
    os.environ["WANDB_RUN_ID"] = str(state.wandbid)

  # init mcmc chain
  def model_logdensity(p, x):
    return 2*model.apply(p, x, False)[1][:, None]
  kernel = mcmc.get_RWMH_kernel(config, model_logdensity)
  kernel = jax.pmap(kernel, in_axes=(0,0,0,None))

  step_fn = tutils.get_train_step(config.train, loss_fn, optimizer)
  sharded_key = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
  samples = state.samples
  running_energy = []
  running_var = []
  for step in range(init_step, config.train.n_time_steps + 1):

    # update model
    sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
    state, loss, aux = step_fn(subkeys, state, samples)
    state = state.replace(step=step)

    # run mcmc
    for _ in range(config.train.mcmc_iter_per_step):
      sharded_key, subkeys = kfac_jax.utils.p_split(sharded_key)
      samples, AR = kernel(subkeys, state.model_params, samples, state.sigma)
      state = state.replace(sigma=mcmc.update_sigma(AR, state.sigma))

    # log
    if jax.process_index() == 0:
      running_energy.append(aux.energy_loc.mean().item())
      running_var.append(aux.energy_var.mean().item())
      if len(running_energy) > 1000:
        running_energy.pop(0)
        running_var.pop(0)
      smoothed_energy = np.mean(np.sort(np.array(running_energy))[50:-50])
      smoothed_std = np.mean(np.sort(np.sqrt(np.array(running_var)))[50:-50])
      wandb.log(dict(AR=AR.mean().item(), 
        sigma=state.sigma,
        energy_smoothed=np.abs(smoothed_energy-molecule.gs_energy)/np.abs(molecule.gs_energy),
        energy_rel=np.abs(aux.energy_loc.mean().item()-molecule.gs_energy)/np.abs(molecule.gs_energy),
        energy_abs=aux.energy_loc.mean().item()-molecule.gs_energy,
        energy_var=aux.energy_var.mean().item(),
        std_smoothed=smoothed_std,
        grad_norm=aux.grad_norm.mean().item(),
        loss=loss.mean().item()
      ))

    if (step % config.train.save_every == 0) and (jax.process_index() == 0):
      saved_state = state.replace(key=sharded_key[0])
      checkpoints.save_checkpoint(checkpoint_dir, saved_state, 
                                  step=step//config.train.save_every, 
                                  prefix='model_',
                                  keep_every_n_steps=5,
                                  overwrite=True)
