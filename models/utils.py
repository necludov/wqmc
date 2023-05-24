import flax
import optax
import functools
import jax.numpy as jnp
import jax
import os
import numpy as np

from systems_catalog import system_catalog
from models.psiformer import make_fermi_net

_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]

from models.ferminet import networks

def init_model(key, config):
  molecule = system_catalog[config.dim][config.system]
  nspins = [molecule.n_up_electrons, molecule.n_down_electrons]
  model = make_fermi_net(nspins=nspins,
                         charges=molecule.nuclei_charge,
                         determinants=config.model.n_dets,
                         bias_orbitals=False,
                         rescale_inputs=True,
                         num_layers=config.model.num_layers,
                         num_heads=config.model.n_attention_heads,
                         heads_dim=64,
                         mlp_hidden_dims=(config.model.num_hidden,),
                         use_layer_norm=False)
  initial_params = model.init(key)
  model.apply = jax.vmap(model.apply, in_axes=[None, 0, None, None, None, None])
  model.orbitals = jax.vmap(model.orbitals, in_axes=[None, 0, None, None, None, None])
  spins = jnp.concatenate([jnp.ones([molecule.n_up_electrons]), -jnp.ones([molecule.n_down_electrons])])
  model_partial = networks.Network(
      options=model.options,
      init=model.init,
      apply=lambda p, x, jastrow: model.apply(p, x, spins, molecule.nuclei_position, molecule.nuclei_charge, jastrow),
      orbitals=lambda p, x, jastrow: model.orbitals(p, x, spins, molecule.nuclei_position, molecule.nuclei_charge, jastrow)
  )
  return model_partial, initial_params
