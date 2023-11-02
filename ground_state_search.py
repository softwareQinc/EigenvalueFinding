from math import sqrt
import matplotlib as plt
from eig_search import *
import numpy as np
from random import random
from scipy.stats import unitary_group
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector, partial_trace


def get_ground_state(matrix, epsilon, theta0=None, above_half=False):
    """PURPOSE: Prepare mixed state whose overlap with  the true ground state is high
    INPUT: -Matrix matrix, desired error epsilon,
    Recall that the first part of the algorithm is to produce an estimate theta0 of lambda0. To speed things up, we optionally allow the user to cheat and supply such an estimate.
    We also allow the user to ask for the us to ignore eigenvalues less than 0.5. NOTE: It is not clear whether this is useful now that we know that our trick for Hermiticizing non-Hermitian matrices doesn't work...
    OUTPUT: Density matrix
    """
    ef = EigenvalueFinding(matrix, epsilon / 4, above_half=above_half)
    # print("Number of qubits is roughly", ef.qpe_bits)
    # Step 1: Get estimate \theta_0
    if theta0 is None:
        theta0 = find_min(ef)[-1]
        # print("The estimated value for the minimum eigenvalue is", theta0)

    # Step 2: Construct the Grover circuit
    # This part is very tricky. We need to explicitly construct the Grover oracle
    oracle_list = [0] * (2 ** (
            ef.qpe_bits + ef.n))  # The entry at index x will be 1 iff x is close to theta_0. This is required by Qiskit
    good_states = []  # We will also build a list of good states
    for x in range(2 ** ef.qpe_bits):
        if abs(x / 2 ** ef.qpe_bits - theta0) < epsilon / 2:  # If x is close to theta_0
            # We can't just mark |x>, but rather, we need to mark all states of the form |y>|x>, where |y> is what's inside the cloc register
            # Thus we iterate over all bit strings y:
            for y in range(2 ** ef.n):
                # ... and add them to the list of good states AND modify the oracle list
                z = bin(y)[2:].zfill(ef.n) + bin(x)[2:].zfill(ef.qpe_bits)[::-1]
                good_states.append(z)
                oracle_list[int(z, 2)] = 1

    problem = AmplificationProblem(oracle=Statevector(oracle_list),
                                   state_preparation=algorithm_a(ef),
                                   is_good_state=good_states)
    grover = Grover(sampler=Sampler())

    # Step 3: Construct Grover circuit
    # We will postselect later, so it really doesn't matter how many iterations we choose. I have selected 2 so that it is a nontrivial number, but also doesn't take too long
    qc = grover.construct_circuit(problem=problem, power=2, measurement=False)
    # Step 4: Execute circuit and extract statevector
    # In order to postselect, we need to append an ancilla qubit and create a bit oracle.
    # Concretely, we create the state |bad>|0> + |good>|1> and postselect on |1> in the second register
    # Recall that a bit oracle is (I\otimes H) (controlled-phase oracle) (I\otimes H), with ctrl = last qubit
    qc.add_register(QuantumRegister(1))
    qc.h(ef.qpe_bits + ef.n)

    # Now construct phase oracle and controlled phase oracle (po and cpo)
    po = QuantumCircuit(ef.qpe_bits + ef.n)
    po.diagonal(
        [(-1) ** (bin(z)[2:].zfill(ef.qpe_bits + ef.n) in good_states) for z in range(2 ** (ef.qpe_bits + ef.n))],
        list(range(ef.qpe_bits + ef.n)))
    cpo = po.to_gate().control(num_ctrl_qubits=1, label="CPO")
    qc.append(cpo, [ef.qpe_bits + ef.n] + list(range(ef.qpe_bits + ef.n)))  # Need to control/target the right qubits

    qc.h(ef.qpe_bits + ef.n)
    backend = Aer.get_backend("statevector_simulator")
    qc = transpile(qc, backend)
    job = backend.run(qc)
    result = job.result()
    svec = result.get_statevector(qc)

    svec = svec.evolve(Statevector([0, 1]).to_operator(), [ef.qpe_bits + ef.n])  # Postselect last qubit being |1>
    # ^Here, the Statevector([0, 1]).to_operator() bit creates |1><1|

    # normalize
    svec = svec / np.linalg.norm(svec)

    # Now trace out the QPE clock qubits and the cpo qubit (that was just postselected)
    return theta0, partial_trace(svec, list(range(ef.qpe_bits)) + [ef.qpe_bits + ef.n])


if __name__ == "__main__":
    # Specify error and matrix size
    error = 2 ** (-3)
    dim = 2 ** 4
    print("Running eigenvalue search on random {0}x{0} matrix with error = {1} ".format(dim, error))

    # Get some random eigs
    d = np.sort([error + (1 - 2 * error) * random() for _ in range(dim)])
    print("Eigs are", d)
    un = unitary_group.rvs(dim)
    mat = un @ np.diag(d) @ un.conj().T
    _, p = np.linalg.eigh(mat)  # _ is just d
    # psi_0 = p[:, 0]

    theta0_error = (0.5 - random()) * error / 4  # Choose a random amount for theta0 to be off by
    _, rho = get_ground_state(mat, error, theta0=d[0] + theta0_error)

    good_projector = sum(Statevector(p[:, i]).to_operator() for i in range(dim) if d[i] - d[0] < error)
    overlap = rho.evolve(good_projector).trace()
    print("Overlap is", overlap.real)  # Ignore small imaginary component coming from roundoff error
