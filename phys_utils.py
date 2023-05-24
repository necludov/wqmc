import numpy as np

import jax
import jax.numpy as jnp
from systems_catalog import system_catalog
from systems_catalog import get_pyscf_molecule
from models import utils as mutils


# hartree units
hbar = 1.0 # reduced Planck constant
m_e = 1.0 # electron mass
e = 1.0 # electron charge
a0 = 1.0 # Bohr radius
eps0 = a0*m_e*e**2/(4*jnp.pi*hbar**2)


def get_potential(config):

  molecule = system_catalog[config.dim][config.system]
  nuclei_x = molecule.nuclei_position
  nuclei_c = molecule.nuclei_charge
  n_electrons = molecule.n_up_electrons + molecule.n_down_electrons
  
  def dist(X, Y, remove_diag=False):
    # X.shape, Y.shape = [n_x, dim], [n_y, dim]
    diff = X[:, None, ...] - Y[None, ...]
    if remove_diag:
      n = diff.shape[0]
      return jnp.sqrt(((diff + jnp.eye(n)[..., None])**2).sum(2))*(1.0 - jnp.eye(n))
    return jnp.sqrt((diff**2).sum(2))

  def V(x):
    x = x.reshape((n_electrons,config.dim))
    en_inter = dist(x, nuclei_x)
    ee_inter = dist(x, x, True)
    ee_inter = ee_inter[jnp.triu_indices_from(ee_inter, 1)]
    nn_inter = dist(nuclei_x, nuclei_x, True)
    nn_inter = nn_inter/nuclei_c.reshape((-1,1))/nuclei_c.reshape((1,-1))
    nn_inter = nn_inter[jnp.triu_indices_from(nn_inter, 1)]
    potential = (1./ee_inter).sum() + (1./nn_inter).sum() - (nuclei_c.reshape(1,-1)/en_inter).sum()
    return potential

  return jax.vmap(V)


import pyscf

def get_init_psi(config):
  molecule = system_catalog[config.dim][config.system]
  n_electrons = (molecule.n_up_electrons, molecule.n_down_electrons)

  mol = get_pyscf_molecule(config.system, molecule)
  mf = pyscf.scf.RHF(mol)
  mf.kernel()
  coeffs = (mf.mo_coeff,)
  gto_op = 'GTOval_sph'

  def get_matrices(x):
    # x.shape = (bs, n_electrons*dim)
    bs = x.shape[0]
    x = x.reshape((-1, config.dim))
    ao_values = mol.eval_gto(gto_op, x)
    mos = tuple(np.matmul(ao_values, coeff) for coeff in coeffs)
    mos *= 2
    mos = [np.reshape(mo, (bs, np.sum(n_electrons), -1)) for mo in mos]
    alpha_spin = mos[0][..., :n_electrons[0], :n_electrons[0]]
    beta_spin = mos[1][..., n_electrons[0]:, :n_electrons[1]]
    return alpha_spin, beta_spin

  def get_logp(x):
    matrices = get_matrices(x)
    slogdets = [np.linalg.slogdet(elem) for elem in matrices]
    sign_alpha, sign_beta = [elem[0] for elem in slogdets]
    log_abs_wf_alpha, log_abs_wf_beta = [elem[1] for elem in slogdets]
    log_abs_slater_determinant = log_abs_wf_alpha + log_abs_wf_beta
    sign = sign_alpha * sign_beta
    return 2*log_abs_slater_determinant.reshape((-1,1))

  def get_orbitals(x):
    target = get_matrices(x)
    ndet = target[0].shape[0]
    na = target[0].shape[1]
    nb = target[1].shape[1]
    target = jnp.concatenate(
        (jnp.concatenate((target[0], jnp.zeros((ndet, na, nb))), axis=-1),
          jnp.concatenate((jnp.zeros((ndet, nb, na)), target[1]), axis=-1)),
        axis=-2)
    return target

  return get_logp, get_orbitals
