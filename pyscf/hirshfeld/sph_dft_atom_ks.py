#!/usr/bin/env python
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#         Susi Lehtola <susi.lehtola@gmail.com>
# Original file see https://github.com/pyscf/pyscf/pull/1143/files
# Slightly modified for hirshfeld volume evaluation

import numpy as np
from pyscf import gto, scf, dft
from pyscf.lib import logger
from pyscf.scf import atom_hf, atom_hf_pp, ADIIS
from pyscf.dft import rks
from pyscf.data import elements
from scipy.interpolate import make_interp_spline

from pyscf.hirshfeld.sph_dft_elements import NRSRHFS_CONFIGURATION


def get_atm_nrks(mol, atomic_configuration=NRSRHFS_CONFIGURATION, xc='slater', grid=(120, 770), basis="aug-cc-pVQZ", pseudo=None):
    if isinstance(mol, scf.hf.SCF):
        xc = getattr(mol, "xc", "HF")
        mol_obj = mol.mol
        if pseudo is None:
            pseudo = getattr(mol_obj, 'pseudo', None)
        mol = mol_obj
    elem_list = set([a[0] for a in mol._atom])
    logger.info(mol, 'Spherically averaged atomic KS for %s', elem_list)

    atm_scf_result = {}
    for elem in elem_list:
        elem_chrg = elements.charge(elem)

        if pseudo:
            # Build atom with pseudopotential — nelectron will be
            # the valence count (e.g. 4 for Si with gth-pbe).
            # GTH PPs always remove an even number of core electrons,
            # so elem_chrg % 2 matches valence electron parity.
            atm = gto.Mole(atom=elem, basis=basis, pseudo=pseudo,
                           verbose=mol.verbose)
            atm.spin = elem_chrg % 2   # parity-safe for build()
            atm.build()
            atm.spin = atm.nelectron   # all valence e⁻ unpaired
            atm.a = None               # required by PP integral helpers
        else:
            atm = gto.Mole(atom=elem, basis=basis, verbose=mol.verbose,
                           spin=elem_chrg).build()

        nao = atm.nao
        # nao == 0 for the case that no basis was assigned to an atom
        if nao == 0 or atm.nelectron == 0:  # GHOST
            raise ValueError("Ghost atom not implemented!")

        if pseudo:
            atm_ks = AtomSphAverageRKSPP(atm)
        else:
            atm_ks = AtomSphericAverageRKS(atm)
        atm_ks.atomic_configuration = atomic_configuration
        atm_ks.xc = xc
        atm_ks.grids.atom_grid = grid
        atm_ks.verbose = mol.verbose
        my_diis_obj = ADIIS()
        my_diis_obj.space = 12
        atm_ks.diis = my_diis_obj
        atm_ks.run()
        atm_scf_result[elem] = atm_ks
    return atm_scf_result


class AtomSphAverageRKS(rks.RKS, atom_hf.AtomSphericAverageRHF):
    def __init__(self, mol, *args, **kwargs):
        atom_hf.AtomSphericAverageRHF.__init__(self, mol)
        rks.RKS.__init__(self, mol, *args, **kwargs)

        # SAP guess is perfect for atoms
        self.init_guess = 'vsap'

    eig = atom_hf.AtomSphericAverageRHF.eig
    get_occ = atom_hf.AtomSphericAverageRHF.get_occ
    get_grad = atom_hf.AtomSphericAverageRHF.get_grad


AtomSphericAverageRKS = AtomSphAverageRKS


class AtomSphAverageRKSPP(AtomSphAverageRKS):
    """Spherically averaged atomic RKS with pseudopotential support."""
    def __init__(self, mol, *args, **kwargs):
        super().__init__(mol, *args, **kwargs)
        # VSAP initial guess does not support pseudopotentials
        self.init_guess = '1e'

    def get_hcore(self, mol=None):
        if mol is None:
            mol = self.mol
        h = mol.intor('int1e_kin', hermi=1)
        h += atom_hf_pp.get_pp_nl(mol)
        h += atom_hf_pp.get_pp_loc(mol)
        return h


def spline_radial(x, y, k=3):
    # log sampled spline interpolation
    spl = make_interp_spline(np.log(x), y, k=k)

    def f(x_in):
        y_out = spl(np.log(x_in))
        y_out[y_out<0] = 0
        return y_out
    return f


def free_atom_info(mf_atom, ngrid=500):
    ni = dft.numint.NumInt()
    r, w = dft.radi.gauss_chebyshev(ngrid)
    rc = np.zeros((len(r), 3))
    rc[:, 0] = r
    rw = 4 * np.pi * w * r**2
    rao = ni.eval_ao(mf_atom.mol, rc)
    rrho = ni.eval_rho(mf_atom.mol, rao, mf_atom.make_rdm1())
    v_free = (rrho * r**3 * rw).sum()
    spl_free = spline_radial(r, rrho)
    return v_free, spl_free


if __name__ == '__main__':
    mol = gto.Mole()
    mol.verbose = 0

    mol.atom = "O; H 1 0.94; H 1 0.94 2 104.5"
    mol.basis = "6-31G"
    mol.build()
    print()
    for k, v in get_atm_nrks(mol, xc="PBE").items():
        print(k, v.e_tot)
