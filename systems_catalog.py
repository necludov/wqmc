import jax.numpy as jnp
from collections import namedtuple


molecule = namedtuple('molecule', ['gs_energy', 'nuclei_position', 'nuclei_charge', 'n_up_electrons', 'n_down_electrons'])


system_catalog = \
    {
        3: {
            'H': molecule(
                gs_energy = -0.5,
                nuclei_position=jnp.array([[0., 0., 0.]]),
                nuclei_charge=jnp.array([1]),
                n_up_electrons=1,
                n_down_electrons=0
            ),
            'He': molecule(
                gs_energy = -2.90372,
                nuclei_position=jnp.array([[0., 0., 0.]]),
                nuclei_charge=jnp.array([2]),
                n_up_electrons=1,
                n_down_electrons=1
            ),
            'H2': molecule(
                gs_energy = -1.1744477,
                nuclei_position=jnp.array([[-0.7, 0., 0.],
                                           [0.7, 0., 0.]]),
                nuclei_charge=jnp.array([1,
                                         1]),
                n_up_electrons=1,
                n_down_electrons=1
            ),
            'Li': molecule(
                gs_energy = -7.47806032,
                nuclei_position=jnp.array([[0.0, 0., 0.]]),
                nuclei_charge=jnp.array([3]),
                n_up_electrons=2,
                n_down_electrons=1
            ),
            'Be': molecule(
                gs_energy = -14.66736,
                nuclei_position=jnp.array([[0.0, 0., 0.]]),
                nuclei_charge=jnp.array([4]),
                n_up_electrons=2,
                n_down_electrons=2
            ),
            'B': molecule(
                gs_energy = -24.65391,
                nuclei_position=jnp.array([[0., 0., 0.]]),
                nuclei_charge=jnp.array([5]),
                n_up_electrons=2,
                n_down_electrons=3
            ),
            'C': molecule(
                gs_energy = -37.8450,
                nuclei_position=jnp.array([[0., 0., 0.]]),
                nuclei_charge=jnp.array([6]),
                n_up_electrons=2,
                n_down_electrons=4
            ),
            'Li2': molecule(
                gs_energy = -14.9954,
                nuclei_position=jnp.array([[-5.051/2, 0., 0.],
                    [5.051/2, 0., 0.]
                ]),
                nuclei_charge=jnp.array([3,3]),
                n_up_electrons=3,
                n_down_electrons=3
            ),
            'N': molecule(
                gs_energy = -54.5892,
                nuclei_position=jnp.array([[0., 0., 0.]]),
                nuclei_charge=jnp.array([7]),
                n_up_electrons=2,
                n_down_electrons=5
            ),
            'Ne': molecule(
                gs_energy = -128.9376,
                nuclei_position=jnp.array([[0., 0., 0.]]),
                nuclei_charge=jnp.array([10]),
                n_up_electrons=5,
                n_down_electrons=5
            ),
            'H10': molecule(
                gs_energy = -5.6655,
                nuclei_position=jnp.array([[0.0, 0.0, 0.0],
                    [0.95305/0.52917721092, 0.0, 0.0],
                    [1.9061/0.52917721092, 0.0, 0.0],
                    [2.85914/0.52917721092, 0.0, 0.0],
                    [3.81219/0.52917721092, 0.0, 0.0],
                    [4.76524/0.52917721092, 0.0, 0.0],
                    [5.71829/0.52917721092, 0.0, 0.0],
                    [6.67134/0.52917721092, 0.0, 0.0],
                    [7.62439/0.52917721092, 0.0, 0.0],
                    [8.57743/0.52917721092, 0.0, 0.0],
                ]),
                nuclei_charge=jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
                n_up_electrons=5,
                n_down_electrons=5
            )
        }

    }

import pyscf
import re
import functools

def get_pyscf_molecule(system, molecule):
    system = system.split(',')
    symbols = list(map(lambda s: s[:len(s)-len(re.findall('\d+', s))], system))
    numbers = list(map(lambda s: 1 if len(s) == 0 else int(s[0]), map(lambda s: re.findall('\d+', s), system)))
    symbols = list(map(lambda s,n: [s]*n, symbols, numbers))
    symbols = functools.reduce(lambda s1, s2: s1+s2, symbols)
    strings = list(map(lambda s, coords: [s] + coords, symbols, molecule.nuclei_position.tolist()))
    strings = list(map(lambda s: functools.reduce(lambda s1, s2: str(s1)+' '+str(s2), s), strings))
    strings = '; '.join(strings)
    print(strings)
    mol = pyscf.M(
        atom = strings,
        basis = 'sto-6g',
        spin = molecule.n_up_electrons - molecule.n_down_electrons,
        unit='bohr'
    )
    return mol
