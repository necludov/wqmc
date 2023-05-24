# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multiplicative envelope functions."""

import enum
from typing import Any, Mapping, Sequence, Union, Tuple

import attr
import jax
import jax.numpy as jnp
from typing_extensions import Protocol

_MAX_POLY_ORDER = 5  # highest polynomial used in envelopes


class EnvelopeType(enum.Enum):
  """The point at which the envelope is applied."""
  PRE_ORBITAL = enum.auto()
  PRE_DETERMINANT = enum.auto()


class EnvelopeLabel(enum.Enum):
  """Available multiplicative envelope functions."""
  ISOTROPIC = enum.auto()
  DIAGONAL = enum.auto()
  FULL = enum.auto()
  NULL = enum.auto()
  STO = enum.auto()
  STO_POLY = enum.auto()


class EnvelopeInit(Protocol):

  def __call__(
      self, natom: int, output_dims: Union[int, Sequence[int]], ndim: int
  ) -> Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
    """Returns the envelope parameters.

    Envelopes applied separately to each spin channel must create a sequence of
    parameters, one for each spin channel. Other envelope types must create a
    single mapping.

    Args:
      natom: Number of atoms in the system.
      output_dims: The dimension of the layer to which the envelope is applied,
        per-spin channel for pre_determinant envelopes and a scalar otherwise.
      ndim: Dimension of system. Change with care.
    """


class EnvelopeApply(Protocol):

  def __call__(self, *, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
               **kwargs: jnp.ndarray) -> jnp.ndarray:
    """Returns a multiplicative envelope to ensure boundary conditions are met.

    If the envelope is applied before orbital shaping or after determinant
    evaluation, the envelope function is called once and N is the number of
    electrons. If the envelope is applied after orbital shaping and before
    determinant evaluation, the envelope function is called once per spin
    channel and N is the number of electrons in the spin channel.

    Args:
      ae: atom-electron vectors, shape (N, natom, ndim).
      r_ae: atom-electron distances, shape (N, natom, 1).
      r_ee: electron-electron distances, shape (N, nel, 1).
      **kwargs: learnable parameters of the envelope function.
    """


@attr.s(auto_attribs=True)
class Envelope:
  apply_type: EnvelopeType
  init: EnvelopeInit
  apply: EnvelopeApply


def _apply_covariance(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Equivalent to jnp.einsum('ijk,kmjn->ijmn', x, y)."""
  # We can avoid first reshape - just make params['sigma'] rank 3
  i, _, _ = x.shape
  k, m, j, n = y.shape
  x = x.transpose((1, 0, 2))
  y = y.transpose((2, 0, 1, 3)).reshape((j, k, m * n))
  vdot = jax.vmap(jnp.dot, (0, 0))
  return vdot(x, y).reshape((j, i, m, n)).transpose((1, 0, 2, 3))


def make_isotropic_envelope(nspins: Tuple[int, ...], charges: jnp.ndarray, ndim: int) -> Envelope:
  """Creates an isotropic exponentially decaying multiplicative envelope."""

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del ndim  # unused
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.ones(shape=(natom, output_dim))
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes an isotropic exponentially-decaying multiplicative envelope."""
    del ae, r_ee  # unused
    # return jnp.exp(-2.0/(ndim - 1.0) * jnp.sum(r_ae * charges.reshape((1,-1,1)))/r_ae.shape[0])
    return jnp.sum(jnp.exp(-r_ae * sigma) * pi, axis=1)

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def make_diagonal_envelope() -> Envelope:
  """Creates a diagonal exponentially-decaying multiplicative envelope."""

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    params = []
    for output_dim in output_dims:
      params.append({
          'pi': jnp.ones(shape=(natom, output_dim)),
          'sigma': jnp.ones(shape=(natom, ndim, output_dim))
      })
    return params

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray, r_ee: jnp.ndarray,
            pi: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Computes a diagonal exponentially-decaying multiplicative envelope."""
    del r_ae, r_ee  # unused
    r_ae_sigma = jnp.linalg.norm(ae[..., None] * sigma, axis=2)
    return jnp.sum(jnp.exp(-r_ae_sigma) * pi, axis=1)

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def make_null_envelope() -> Envelope:
  """Creates an no-op (identity) envelope."""

  def init(
      natom: int, output_dims: Sequence[int], ndim: int = 3
  ) -> Sequence[Mapping[str, jnp.ndarray]]:
    del natom, ndim  # unused
    return [{} for _ in output_dims]

  def apply(*, ae: jnp.ndarray, r_ae: jnp.ndarray,
            r_ee: jnp.ndarray) -> jnp.ndarray:
    del ae, r_ae, r_ee
    return jnp.ones(shape=(1,))

  return Envelope(EnvelopeType.PRE_DETERMINANT, init, apply)


def get_envelope(
    envelope_label: EnvelopeLabel,
    **kwargs: Any,
) -> Envelope:
  """Gets the desired multiplicative envelope function.

  Args:
    envelope_label: envelope function required.
    **kwargs: keyword arguments forwarded to the envelope.

  Returns:
    (envelope_type, envelope), where envelope_type describes when the envelope
    should be applied in the network and envelope is the envelope function.
  """
  envelope_builders = {
      EnvelopeLabel.STO: make_sto_envelope,
      EnvelopeLabel.STO_POLY: make_sto_poly_envelope,
      EnvelopeLabel.ISOTROPIC: make_isotropic_envelope,
      EnvelopeLabel.DIAGONAL: make_diagonal_envelope,
      EnvelopeLabel.FULL: make_full_envelope,
      EnvelopeLabel.NULL: make_null_envelope,
  }
  return envelope_builders[envelope_label](**kwargs)
