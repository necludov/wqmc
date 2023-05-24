from typing import Any
import functools

import jax
import jax.numpy as jnp
import flax
import optax
import flax.jax_utils as flax_utils
import kfac_jax

from models import utils as mutils
from models.ferminet import curvature_tags_and_blocks

@flax.struct.dataclass
class QMCState:
  step: int
  opt_state: Any
  model_params: Any
  key: Any
  wandbid: Any
  sigma: float
  samples: jnp.array

@flax.struct.dataclass
class AuxData:
  energy_loc: jnp.array
  energy_var: jnp.array
  next_samples: jnp.array
  grad_norm: jnp.array


def get_train_step(config, loss_fn, optimizer):
  if isinstance(optimizer, optax.GradientTransformation):
    train_step = get_train_step_optax(config, loss_fn, optimizer)
  elif isinstance(optimizer, kfac_jax.Optimizer):
    train_step = get_train_step_kfac(config, loss_fn, optimizer)
  else:
    raise ValueError(f'Unknown optimizer: {optimizer}')
  return train_step


def get_train_step_optax(config, loss_fn, optimizer):
  
  @functools.partial(jax.pmap, axis_name='batch', in_axes=(0,0,0,0,None))
  def train_step_optax(key, params, opt_state, batch, step):
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    (loss, aux), grad = grad_fn(key, params, batch, step)
    grad = jax.lax.pmean(grad, axis_name='batch')
    updates, opt_state = optimizer.update(grad, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    loss = jax.lax.pmean(loss, axis_name='batch')
    return new_params, opt_state, loss, aux

  def train_step(sharded_key, qmc_state, batch):
    new_params, opt_state, loss, aux = train_step_optax(sharded_key, 
      qmc_state.model_params, 
      qmc_state.opt_state, 
      batch,
      qmc_state.step
    )
    qmc_state = qmc_state.replace(model_params=new_params, opt_state=opt_state)
    return qmc_state, loss, aux

  return train_step

def get_train_step_kfac(config, loss_fn, optimizer):

  def train_step_kfac(sharded_key, qmc_state, batch):
    shared_mom = kfac_jax.utils.replicate_all_local_devices(jnp.zeros([]))
    shared_damping = kfac_jax.utils.replicate_all_local_devices(jnp.asarray(1e-3))
    next_params, next_opt_state, stats = optimizer.step(
      params=qmc_state.model_params,
      state=qmc_state.opt_state,
      rng=sharded_key,
      batch=batch,
      momentum=shared_mom,
      damping=shared_damping,
      global_step_int=qmc_state.step
    )
    qmc_state = qmc_state.replace(model_params=next_params, opt_state=next_opt_state)
    return qmc_state, stats['loss'], stats['aux']
  
  return train_step_kfac


def get_optimizer(config, loss_fn, key, params, samples):
  if config.optimizer == 'kfac':
    optimizer = get_optimizer_kfac(config, loss_fn)
    subkeys = kfac_jax.utils.make_different_rng_key_on_all_devices(key)
    opt_state = optimizer.init(flax_utils.replicate(params), subkeys, samples)
    return optimizer, opt_state
  else:
    optimizer = get_optimizer_optax(config)
    opt_state = optimizer.init(params)
    return optimizer, flax_utils.replicate(opt_state)

def get_optimizer_optax(config):
  schedule = optax.exponential_decay(config.lr, 10_000, 3)
  if config.optimizer == 'adam':
    optimizer = optax.adam(learning_rate=schedule, b1=config.beta1, b2=config.beta2, eps=config.eps)
  elif config.optimizer == 'yogi':
    optimizer = optax.yogi(learning_rate=schedule, b1=config.beta1, b2=config.beta2, eps=config.eps)
  elif config.optimizer == 'adabelief':
    optimizer = optax.adabelief(learning_rate=schedule, b1=config.beta1, b2=config.beta2, eps=config.eps)
  elif config.optimizer == 'lamb':
    optimizer = optax.lamb(learning_rate=schedule, b1=config.beta1, eps=config.eps)

  optimizer = optax.chain(
    optax.clip(config.grad_clip),
    optimizer
  )
  return optimizer

def get_optimizer_kfac(config, loss_fn):
  # learning_rate_schedule = lambda _t: jnp.array([config.lr])
  learning_rate_schedule = lambda _t: config.lr/(1.0 + 2*(_t/config.n_time_steps))
  optimizer = kfac_jax.Optimizer(
        jax.value_and_grad(loss_fn, argnums=0, has_aux=True),
        l2_reg=0.0,
        norm_constraint=1e-2,
        value_func_has_aux=True,
        value_func_has_rng=True,
        learning_rate_schedule=learning_rate_schedule,
        curvature_ema=0.95,
        inverse_update_period=1,
        min_damping=1.e-4,
        num_burnin_steps=0,
        register_only_generic=False,
        estimation_mode='fisher_exact',
        multi_device=True,
        pmap_axis_name='batch',
        auto_register_kwargs=dict(
            graph_patterns=curvature_tags_and_blocks.GRAPH_PATTERNS,
        ),
        # debug=True
  )
  return optimizer


def get_pretrain_loss(config, model):

  def l2_loss(key, params, batch, step):
    samples, target = batch
    loss = ((model.orbitals(params, samples, True)[0] - target[:, None, ...])**2).sum(1)
    return loss.mean(), None

  return l2_loss

def get_loss(config, model, V):
  if config.train.optimizer == 'kfac':
    if config.method == 'qvmc':
      return get_qvmc_kfac_loss(config, model, V)
    elif config.method == 'w2':
      return get_w2_kfac_loss(config, model, V)
  if config.method == 'qvmc':
    return get_qvmc_train_loss(config, model, V)
  elif config.method == 'w2':
    return get_w2_train_loss(config, model, V)
  raise ValueError(f'unrecognised method: {config.method}')

def get_w2_train_loss(config, model, V):

  log_q = lambda _p, _x: 2*model.apply(_p, _x, False)[1]

  def proj_loss(key, params, batch, step):
    def e_fn(x):
      score = jax.grad(lambda _x: log_q(params, _x).sum())
      divscore = jax.jacrev(lambda _x: score(_x).sum(0))(x).trace(axis1=0,axis2=2)
      score_val = score(x)
      energy = V(x) - 0.25*divscore - 0.125*(score_val**2).sum(1)
      return energy.sum(), (energy, score_val)

    def filter(data):
      median = jnp.median(jax.lax.all_gather(data, 'batch'))
      deviation = jnp.mean(jnp.abs(data - median))
      return jnp.clip(data, median - 5*deviation, median + 5*deviation)

    dt = config.train.dt
    
    de_fn = jax.value_and_grad(e_fn, has_aux=True)
    (energy, (energy_loc, score_val)), de = de_fn(batch)
    de = jax.lax.stop_gradient(de)
    de_norm = jnp.linalg.norm(de, axis=-1, keepdims=True)
    # max_norm = jnp.sqrt(de.shape[1]/3)
    # de = jnp.where(de_norm < max_norm, de, max_norm*de/de_norm)
    de = jnp.tanh(de)
    next_batch = batch - dt*de
    next_batch = jax.lax.stop_gradient(next_batch)
    
    score_norm = jax.lax.stop_gradient(jnp.linalg.norm(score_val, axis=-1, keepdims=True))
    median = jnp.median(jax.lax.all_gather(score_norm, 'batch'))
    deviation = jnp.mean(jnp.abs(score_norm - median))
    mask = score_norm < (median + 5*deviation)
    score_val = score_val*mask
    loss = (de*score_val).sum(1)
    loss = loss*(len(mask)/mask.sum())
    print(loss.shape, 'loss.shape', flush=True)

    energy_mean = jax.lax.pmean(jnp.mean(energy_loc), 'batch')
    energy_var = jnp.mean((energy_mean-energy_loc)**2)
    aux = AuxData(energy_loc=energy_loc, energy_var=energy_var, next_samples=next_batch, grad_norm=jnp.median(de_norm))
    return loss.mean(), aux

  return proj_loss


def get_qvmc_train_loss(config, model, V):
  rho = 5.0

  def e_fn(x, params):
    score = jax.grad(lambda _x: 2*model.apply(params, _x, False)[1].sum())
    divscore = jax.jacrev(lambda _x: score(_x).sum(0))(x).trace(axis1=0,axis2=2)
    energy = V(x) - 0.25*divscore - 0.125*(score(x)**2).sum(1)
    return energy

  def qvmc_loss(key, params, batch, step):
    energy_loc = jax.lax.stop_gradient(e_fn(batch, params))
    median = jnp.median(jax.lax.all_gather(energy_loc, 'batch'))
    deviation = jnp.mean(jnp.abs(energy_loc - median))
    energy_clipped = jnp.clip(energy_loc, median - rho*deviation, median + rho*deviation)
    loss = (energy_clipped - energy_clipped.mean())*2*model.apply(params, batch, False)[1]

    energy_mean = jax.lax.pmean(jnp.mean(energy_loc), 'batch')
    energy_var = jax.lax.pmean(jnp.mean((energy_mean-energy_loc)**2), 'batch')
    aux = AuxData(energy_loc=energy_loc, energy_var=energy_var, next_samples=batch, grad_norm=jnp.ones_like(energy_loc))
    return loss.mean(), aux

  return qvmc_loss


################################## KFAC ##################################

def get_w2_kfac_loss(config, model, V):
  rho = 5.0
  log_q = lambda _p, _x: 2*model.apply(_p, _x, False)[1]
  score = jax.grad(lambda _p, _x: log_q(_p, _x).sum(), argnums=1)

  def filter(data):
    median = jnp.median(jax.lax.all_gather(data, 'batch'))
    deviation = jnp.mean(jnp.abs(data - median))
    return jnp.clip(data, median - 5*deviation, median + 5*deviation)

  @jax.custom_jvp
  def w2_loss(params, key, batch):
    def e_fn(x):
      score_val = score(params, x)
      divscore = jax.jacrev(lambda _x: score(params, _x).sum(0))(x).trace(axis1=0,axis2=2)
      energy = V(x) - 0.25*divscore - 0.125*(score_val**2).sum(1)
      return energy.sum(), (energy, score_val)

    de_fn = jax.value_and_grad(e_fn, has_aux=True)
    (energy, (energy_loc, score_val)), de = de_fn(batch)
    energy_loc = jax.lax.stop_gradient(energy_loc)

    score_norm = jax.lax.stop_gradient(jnp.linalg.norm(score_val, axis=-1, keepdims=True))
    median = jnp.median(jax.lax.all_gather(score_norm, 'batch'))
    deviation = jnp.mean(jnp.abs(score_norm - median))
    mask = score_norm < (median + 5*deviation)

    de = jax.lax.stop_gradient(de)
    de_norm = jnp.linalg.norm(de, axis=-1, keepdims=True)
    # max_norm = jnp.sqrt(de.shape[1]/3)
    # de = jnp.where(de_norm < max_norm, de, max_norm*de/de_norm)
    de = jnp.tanh(de)
    next_batch = batch - config.train.dt*de
    next_batch = jax.lax.stop_gradient(next_batch)

    loss = (de*score_val*mask).sum(1)*(len(mask)/mask.sum())
    if config.joint:
      clipped_energy = filter(energy_loc)
      loss += (clipped_energy - clipped_energy.mean())*log_q(params, batch)

    energy_mean = jax.lax.pmean(jnp.mean(energy_loc), 'batch')
    energy_var = jax.lax.pmean(jnp.mean((energy_mean-energy_loc)**2), 'batch')
    aux = AuxData(energy_loc=energy_loc, energy_var=energy_var, next_samples=next_batch, grad_norm=jnp.median(de_norm))
    return loss.mean(), aux

  @w2_loss.defjvp
  def w2_loss_jvp(primals, tangents):
    params, key, data = primals
    loss, aux = w2_loss(params, key, data)
    
    def log_q_(params, data):
      out = 2*model.apply(params, data, False)[1]
      kfac_jax.register_normal_predictive_distribution(out[:,None])
      return out.sum()

    score_ = jax.grad(log_q_, argnums=1)

    score_primal, score_tangent = jax.jvp(score_, (primals[0], primals[2]), (tangents[0], tangents[2]))
    
    score_norm = jax.lax.stop_gradient(jnp.linalg.norm(score_primal, axis=-1, keepdims=True))
    median = jnp.median(jax.lax.all_gather(score_norm, 'batch'))
    deviation = jnp.mean(jnp.abs(score_norm - median))
    mask = score_norm < (median + 5*deviation)
    
    primals_out = loss, aux

    batch, next_batch = primals[2], aux.next_samples
    de = (batch - next_batch)/config.train.dt
    q_tangent_out = (de*score_tangent*mask).sum(1)*(len(mask)/mask.sum())

    if config.joint:
      q_primal, q_tangent = jax.jvp(log_q, (primals[0], primals[2]), (tangents[0], tangents[2]))
      clipped_energy = filter(aux.energy_loc)
      q_tangent_out += (clipped_energy - clipped_energy.mean())*q_tangent
    
    tangents_out = (q_tangent_out.mean(), aux)
    return primals_out, tangents_out

  return w2_loss

def get_qvmc_kfac_loss(config, model, V):
  rho = 5.0
  log_q = lambda _p, _x: 2*model.apply(_p, _x, False)[1]

  @jax.custom_jvp
  def qvmc_loss(params, key, batch):
    def e_fn(x):
      score = jax.grad(lambda _x: log_q(params, _x).sum())
      divscore = jax.jacrev(lambda _x: score(_x).sum(0))(x).trace(axis1=0,axis2=2)
      energy = V(x) - 0.25*divscore - 0.125*(score(x)**2).sum(1)
      return energy.sum(), energy

    (_, energy_loc), de = jax.value_and_grad(e_fn, has_aux=True)(batch)
    de_norm = jnp.linalg.norm(de, axis=-1, keepdims=True)
    energy_loc = jax.lax.stop_gradient(energy_loc)

    loss = jax.lax.pmean(jnp.mean(energy_loc), 'batch')
    energy_var = jax.lax.pmean(jnp.mean((loss-energy_loc)**2), 'batch')
    aux = AuxData(energy_loc=energy_loc, energy_var=energy_var, next_samples=batch, grad_norm=jnp.median(de_norm))
    return loss, aux

  @qvmc_loss.defjvp
  def qvmc_loss_jvp(primals, tangents):
    params, key, data = primals
    loss, aux = qvmc_loss(params, key, data)
    median = jnp.median(jax.lax.all_gather(aux.energy_loc, 'batch'))
    deviation = jnp.mean(jnp.abs(aux.energy_loc - median))
    energy_clipped = jnp.clip(aux.energy_loc, median - rho*deviation, median + rho*deviation)
    diff = energy_clipped - jnp.mean(energy_clipped)

    q_primal, q_tangent = jax.jvp(log_q, (primals[0], primals[2]), (tangents[0], tangents[2]))
    
    kfac_jax.register_normal_predictive_distribution(q_primal[:,None])
    primals_out = loss, aux
    device_batch_size = aux.energy_loc.shape[0]
    tangents_out = (jnp.dot(q_tangent, diff) / device_batch_size, aux)
    return primals_out, tangents_out

  return qvmc_loss
