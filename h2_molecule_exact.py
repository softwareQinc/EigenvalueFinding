import openfermion as of
from openfermion.ops import FermionOperator, QubitOperator
from openfermion import MolecularData
import openfermionpyscf as ofpyscf
import numpy as np
from scipy.sparse import linalg
import scipy
import matplotlib.pyplot as plt

basis = 'sto-3g'
multiplicity = 1
bond_length_step = 0.05
min_bond_length = 0.3
max_bond_length = 2.5

hf_energies = []
bond_lengths = np.arange(min_bond_length, max_bond_length, bond_length_step)

for bond_length in bond_lengths:
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
    molecule = MolecularData(geometry, basis, multiplicity, description="")

    # Load data.
    molecule.load()
    # obtain Hamiltonian as an InteractionOperator
    hamiltonian = ofpyscf.generate_molecular_hamiltonian(geometry, basis, multiplicity, charge=0)
    # Convert to a FermionOperator
    hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)
    # Map to QubitOperator using the JWT
    hamiltonian_jw = of.jordan_wigner(hamiltonian_ferm_op)
    # Convert to Scipy sparse matrix
    hamiltonian_jw_sparse = of.get_sparse_operator(hamiltonian_jw)

    # Compute ground state energy
    eigs, _ = scipy.linalg.eig(hamiltonian_jw_sparse.toarray())
    ground_energy = np.min(eigs)

    hf_energies.append(ground_energy)

plt.plot(bond_lengths, hf_energies)
plt.xlabel("Bond length")
plt.ylabel("Ground state energy")
plt.show()

