"""Purpose: Generate exact ground state values for H2 molecule with lots of different bond lengths"""

import openfermion as of
from openfermion import MolecularData
import openfermionpyscf as ofpyscf
import numpy as np
import scipy
import matplotlib.pyplot as plt

basis = 'sto-3g'
multiplicity = 1
bond_length_step = 0.05
min_bond_length = 0.3
max_bond_length = 2.5

hf_energies = []
hf_norms = []
bond_lengths = np.arange(min_bond_length, max_bond_length, bond_length_step)

for bond_length in bond_lengths:
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
    molecule = MolecularData(geometry, basis, multiplicity, description="")

    molecule.load()
    hamiltonian = ofpyscf.generate_molecular_hamiltonian(geometry, basis, multiplicity, charge=0)
    hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)
    hamiltonian_jw = of.jordan_wigner(hamiltonian_ferm_op)
    hamiltonian_jw_sparse = of.get_sparse_operator(hamiltonian_jw)

    # Compute ground state energy
    eigs, _ = [e.real for e in scipy.linalg.eig(hamiltonian_jw_sparse.toarray())]
    ground_energy = np.min(eigs)

    # Compute Hamiltonian norms
    norm = np.max(eigs)+np.abs(ground_energy)

    hf_energies.append(ground_energy)
    hf_norms.append(norm)


print("Hamiltonian norms (min, max): ", [np.min(hf_norms), np.max(hf_norms)])
print("Scaling of the algorithm [sqrt(N) * ||H|| * 1/\epsilon]: ", np.sqrt(2^4) * np.max(hf_norms) * 10^2)


plt.plot(bond_lengths, hf_energies)
plt.xlabel("Bond length")
plt.ylabel("Ground state energy")
plt.show()

