import jax
import jax.numpy as jnp
from jax import random

from systems_catalog import system_catalog

def get_init_samples(key, config):
  molecule = system_catalog[config.dim][config.system]
  n_electrons = molecule.n_up_electrons + molecule.n_down_electrons
  assert n_electrons == molecule.nuclei_charge.sum()
  x = list(map(lambda _x: jnp.array(_x), molecule.nuclei_position.tolist()))
  x = jax.tree_map(lambda _x, _c: jnp.tile(_x, (_c,1)), x, molecule.nuclei_charge.tolist())
  x = jax.lax.concatenate(x, 0).flatten()
  print(x, flush=True)
  samples = x + config.mcmc.init_sigma*random.normal(key, (config.train.batch_size, config.dim*n_electrons))
  return samples
  

def get_RWMH_kernel(config, logp):

  def kernel(key, params, x, sigma=config.mcmc.rwmh_sigma):
    keys = random.split(key)
    proposal = x + sigma*random.normal(keys[0], x.shape)
    u = random.uniform(keys[1], (x.shape[0],1))
    mask = logp(params, proposal) - logp(params, x) > jnp.log(u)
    next_x = mask*proposal + (1-mask)*x
    AR = mask.mean()
    return next_x, AR

  return kernel


def update_sigma(AR, sigma):
  if AR.mean() > 0.55:
    return sigma*1.1
  if AR.mean() < 0.5:
    return sigma/1.1
  return sigma
