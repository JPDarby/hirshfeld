# Carry out Hirshfeld partitioning of the charge density
# https://link.springer.com/content/pdf/10.1007/BF00549096.pdf

import numpy as np

from pyscf import scf, dft, mcscf
from pyscf.hirshfeld.sph_dft_atom_ks import get_atm_nrks, get_atm_nrhf_mf, free_atom_info

class HirshfeldAnalysis:
    """
    This class computes Hirshfeld-partitioned integrals
    over the molecular electronic density.
    """
    mol = None
    mf = None
    xc = None

    result = {}

    def __init__(self, mf, atomic_scf=False):
        self.atomic_scf = atomic_scf
        self.parse_mf(mf)

    def parse_mf(self, mf):
        assert(isinstance(mf, scf.hf.SCF) or
               isinstance(mf, mcscf.casci.CASCI))
        self.mf = mf
        self.mol = mf.mol
        self.xc = mf.xc
        return self

    def perform_free_atom(self):
        result = self.result
        mf = self.mf
        mol = self.mol

        result["mf_elem"] = {}
        result["V_free_elem"] = {}
        result["spl_free_elem"] = {}
        if self.atomic_scf:
            mf_elems = get_atm_nrks(mf, xc=self.xc)
        else:
            mf_elems = get_atm_nrhf_mf(mf)
        for elem in mf_elems:
            mf_elem = mf_elems[elem]
            result["mf_elem"][elem] = mf_elem
            result["V_free_elem"][elem], result["spl_free_elem"][elem] = free_atom_info(mf_elem)

        result["V_free"] = np.zeros(mol.natm)
        for atom in range(mol.natm):
            elem = mol.atom_symbol(atom)
            result["V_free"][atom] = result["V_free_elem"][elem]
        return self

    def perform_hirshfeld(self, fn=None):
        """ Compute self.result object.

            If fn is not None, result['custom']
            will also be filled with an array of integrals, 1 for each atom.

            fn should take r : array (natm, pts, R^3) -> array (natm, pts, S)
            Here, r are coordinates in units of Bohr.

            The return shape from `integrate` will be (atoms,) + S

            The integral done for every atom, a (at r_a), is,

               int fn(r - r_a) chg_a dr^3

            where chg_a = -1*[rho - rho_ref]*(weight function for atom a)
        """

        result = self.result
        mf  = self.mf
        mol = self.mol
        ni  = dft.numint.NumInt()
        grids = getattr(mf, "grids", None)
        if grids is None:
            grids = dft.Grids(mol)
            grids.atom_grid = (77, 302)
            grids.build()

        dm_raw = mf.make_rdm1()

        # Detect spin-polarized calculation
        spin_polarized = isinstance(dm_raw, (tuple, list)) or \
                         (hasattr(dm_raw, 'ndim') and dm_raw.ndim == 3)

        if spin_polarized:
            dm_a, dm_b = dm_raw[0], dm_raw[1]
            dm_total = dm_a + dm_b
            nalpha, nbeta = mol.nelec
        else:
            dm_total = dm_raw

        rho = ni.get_rho(mol, dm_total, grids)
        Ntot = np.vdot(rho, grids.weights)

        # normalize total charge
        rho *= mol.nelectron / Ntot

        # (N_atom x N_gridpoints) array of
        # r - r_atom, where r is a grid point and r_atom is an
        # atom center.
        coords_atoms = grids.coords[None, :, :] - mol.atom_coords()[:, None, :]
        # distance from each atom to each grid point
        rad_atoms = np.linalg.norm(coords_atoms, axis=-1)

        # rho_free = atoms x integration points array of 
        #            "proatom" densities
        rho_free = np.empty((mol.natm, len(grids.coords)))
        for atom in range(mol.natm):
            elem = mol.atom_symbol(atom)
            rho_free[atom] = result["spl_free_elem"][elem](rad_atoms[atom])

        # weights_free = atoms x integration points array of 
        #   percentages -- for partitioning the total el- density.
        #
        # weights_free[i,p] = rho_free[i,p] / sum_j rho_free[j, p]
        tot_free = rho_free.sum(axis=0)
        weights_free = rho_free / (tot_free + (tot_free < 1e-15))

        # electron density partitioned onto every atom
        # rho_eff.sum(axis=0) == rho
        # multiply by grids.weights to integrate numerically
        rho_eff = rho * weights_free

        # integral of r^3 - proxy for atomic volume
        V_eff = (rho_eff * rad_atoms ** 3 * grids.weights).sum(axis=-1)
        # number of electrons on each atom
        elec_eff = (rho_eff * grids.weights).sum(axis=-1)
        elec_eff = np.round(elec_eff, 7)
        # net charge on each atom
        chrg_eff = - elec_eff + mol.atom_charges()
        dipole_eff = - (coords_atoms * rho_eff[:, :, None] * grids.weights[:, None]).sum(axis=-2)

        result["rho_free"] = rho_free
        result["weights_free"] = weights_free
        result["V_eff"] = V_eff
        result["charge_eff"] = chrg_eff
        result["dipole_eff"] = dipole_eff
        result["spin_polarized"] = spin_polarized

        # --- Spin-resolved partitioning ---
        if spin_polarized:
            rho_alpha = ni.get_rho(mol, dm_a, grids)
            rho_beta  = ni.get_rho(mol, dm_b, grids)

            # Normalize each spin channel to its exact electron count  
            Na_int = np.vdot(rho_alpha, grids.weights)
            Nb_int = np.vdot(rho_beta,  grids.weights)
            rho_alpha *= nalpha / Na_int
            rho_beta  *= nbeta  / Nb_int

            # Partition each spin density with the same Hirshfeld weights
            rho_alpha_eff = rho_alpha * weights_free  # (natm, npts)
            rho_beta_eff  = rho_beta  * weights_free

            elec_alpha = (rho_alpha_eff * grids.weights).sum(axis=-1)
            elec_beta  = (rho_beta_eff  * grids.weights).sum(axis=-1)
            elec_alpha = np.round(elec_alpha, 7)
            elec_beta  = np.round(elec_beta, 7)
            
            chrg_eff_alpha = - elec_alpha + mol.atom_charges() * 0.5
            chrg_eff_beta = - elec_beta + mol.atom_charges() * 0.5

            dipole_alpha = - (coords_atoms * rho_alpha_eff[:, :, None]
                              * grids.weights[:, None]).sum(axis=-2)
            dipole_beta  = - (coords_atoms * rho_beta_eff[:, :, None]
                              * grids.weights[:, None]).sum(axis=-2)

            result["charge_eff_alpha"] = chrg_eff_alpha
            result["charge_eff_beta"]  = chrg_eff_beta
            result["dipole_alpha"] = dipole_alpha
            result["dipole_beta"]  = dipole_beta
            # Atomic magnetic moment = N_alpha - N_beta per atom
            result["magmom"] = elec_alpha - elec_beta

        if fn is None:
            result["custom"] = None
        else:
            # normalize total pro-atom charge
            Ntot_free = np.vdot(tot_free, grids.weights)
            tot_free *= mol.nelectron / Ntot_free

            F = fn(coords_atoms)
            assert F.shape[:2] == coords_atoms.shape[:2], \
                        "Invalid return shape from fn."
            ans = np.einsum('ai...,ai,i->a...', F,
                    tot_free*weights_free - rho_eff, grids.weights)
            result["custom"] = ans
        return self

    def run(self, fn=None):
        self.perform_free_atom().perform_hirshfeld(fn)
        return self


if __name__ == '__main__':
    from pyscf import gto
    mol = gto.Mole()
    mol.atom = """
    O 0 0 0; H 0 0 1; H 0 1 0;
    O 0 0 2; H 0 0 3; H 0 1 2
    """
    mol.basis = "cc-pVDZ"
    mol.build()
    mf = dft.RKS(mol, xc="PBE").run()

    anal = HirshfeldAnalysis(mf).run()
    print(anal.result["V_free"])
    print(anal.result["V_eff"])
