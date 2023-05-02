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
bond_length_interval = 0.01
n_points = 250

hf_energies = []
bond_lengths = []

for point in range(3, n_points+1):
    bond_length = 0.3 + bond_length_interval * point
    bond_lengths += [bond_length]
    # description = str(round(bond_length, 3))
    geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
    molecule = MolecularData(geometry, basis, multiplicity, description="")

    # Load data.
    molecule.load()
    # Print out some results of calculation.
    # print('\nAt bond length of {} angstrom, molecular hydrogen has:'.format(bond_length))
    # print('Hartree-Fock energy of {} Hartree.'.format(molecule.hf_energy))

    # Perform electronic structure calculations and
    # obtain Hamiltonian as an InteractionOperator
    hamiltonian = ofpyscf.generate_molecular_hamiltonian(geometry, basis, multiplicity, charge=0)

    # Convert to a FermionOperator
    hamiltonian_ferm_op = of.get_fermion_operator(hamiltonian)

    # Map to QubitOperator using the JWT
    hamiltonian_jw = of.jordan_wigner(hamiltonian_ferm_op)

    # Convert to Scipy sparse matrix
    hamiltonian_jw_sparse = of.get_sparse_operator(hamiltonian_jw)


    # Compute ground energy
    eigs, _ = scipy.linalg.eig(hamiltonian_jw_sparse.toarray())
    ground_energy = np.min(eigs)

    hf_energies.append(ground_energy)

plt.plot(bond_lengths, hf_energies)
plt.show()

