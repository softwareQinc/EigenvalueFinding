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
bond_length_step = 0.05
min_bond_length = 1
max_bond_length = 1

hf_energies = []
bond_lengths = np.arange(min_bond_length, max_bond_length, bond_length_step)

for bond_length in [1.2]:
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
    hamiltonian += 2*np.identity(16)  # Also converts to numpy matrix type
    hamiltonian /= 4

    ef = EigenvalueFinding(hamiltonian, 2**-6)
    energy = find_min(ef)[-1]
    print("Estimated ground state energy is", 4*energy - 2)

    # hf_energies.append(ground_energy)

