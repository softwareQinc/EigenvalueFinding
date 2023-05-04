import openfermion as of
from openfermion.ops import FermionOperator, QubitOperator
from openfermion import MolecularData
import openfermionpyscf as ofpyscf
import numpy as np
from scipy.sparse import linalg
import scipy
import matplotlib.pyplot as plt

from eig_search import EigenvalueFinding, find_min


basis = 'sto-3g'
multiplicity = 1
bond_length_step = 0.01
min_bond_length = 0.3
max_bond_length = 2.5
epsilon = 0.01

estimated_energies = []
exact_energies = []
bond_lengths = np.arange(min_bond_length, max_bond_length, bond_length_step)

for bond_length in bond_lengths:
    # print("Bond length:", bond_length)
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
    hamiltonian = of.get_sparse_operator(hamiltonian_jw)

    # Now shift and rescale:
    shift_factor = 1.1*scipy.linalg.norm(hamiltonian.toarray(), 2)
    hamiltonian += shift_factor*np.identity(16)  # Also converts to numpy matrix type
    hamiltonian /= (2*shift_factor)
    assert all([0<x<1 for x in np.linalg.eigvals(hamiltonian)])
    exact_energy = shift_factor * 2 * np.min(np.linalg.eigvals(hamiltonian)) - shift_factor
    exact_energies.append(exact_energy)

    ef = EigenvalueFinding(hamiltonian, epsilon=epsilon/(2*shift_factor))
    approximate_energy = (shift_factor*2)*find_min(ef)[-1] - shift_factor
    estimated_energies.append(approximate_energy)

    with open("h2_data.txt", "a") as f:
        f.write(str(round(bond_length, 2)) + " " + str(exact_energy.real) + "\n")
#
# plt.plot(bond_lengths, estimated_energies)
# plt.plot(bond_lengths, exact_energies)
# plt.show()
