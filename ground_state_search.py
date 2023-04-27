from math import sqrt
import matplotlib as plt
from eig_search import *
import numpy as np
from random import random
from scipy.stats import unitary_group
import qiskit
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister, ClassicalRegister
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector, partial_trace


def get_ground_state(matrix, epsilon):
    # Step 1: Get estimate \theta_0
    # ef = EigenvalueFinding(matrix, epsilon/4)
    # theta0 = find_min(ef)
    theta0 = 0.21  # Skip step 1 for testing purposes just to save time

    # Step 2: Construct the Grover circuit
    ef = EigenvalueFinding(matrix, epsilon/4)  # Need a bit more precision
    oracle_list = [0]*(2**(ef.qpe_bits + ef.n))  # The entry at index x is 1 iff x is close to theta_0
    good_states = []  # List of good states
    for x in range(2**ef.qpe_bits):
        if abs(x/2**ef.qpe_bits - theta0) < epsilon/2:
            for y in range(2**ef.n):
                z = bin(y)[2:].zfill(ef.n) + bin(x)[2:].zfill(ef.qpe_bits)[::-1]  # Check this...
                good_states.append(z)
                oracle_list[int(z, 2)] = 1

    problem = AmplificationProblem(oracle=Statevector(oracle_list),
                                   state_preparation=algorithm_a(ef),
                                   is_good_state=good_states)
    grover = Grover(sampler=Sampler())

    # Step 3: Figure out how many iterations are needed
    qc = grover.construct_circuit(problem=problem, power=2, measurement=False)
    # Step 4: Execute circuit and extract statevector
    backend = Aer.get_backend("statevector_simulator")
    # First we need to apply the bit oracle in order to postselect
    # Recall that a bit oracle is (I\otimes H) (controlled-phase oracle) (I\otimes H), with ctrl = last qubit
    qc.add_register(QuantumRegister(1))
    qc.h(ef.qpe_bits + ef.n)

    # Now construct phase oracle (po)
    po = QuantumCircuit(ef.qpe_bits + ef.n)

    po.diagonal([(-1)**(bin(z)[2:].zfill(ef.qpe_bits + ef.n) in good_states) for z in range(2**(ef.qpe_bits + ef.n))],
                list(range(ef.qpe_bits + ef.n)))

    cpo = po.to_gate().control(num_ctrl_qubits=1, label="CPO")
    qc.append(cpo, [ef.qpe_bits + ef.n] + list(range(ef.qpe_bits + ef.n)))

    qc.h(ef.qpe_bits + ef.n)
    print(qc)
    qc = transpile(qc, backend)
    job = backend.run(qc)
    result = job.result()
    svec = result.get_statevector(qc)

    svec = svec.evolve(Statevector([0, 1]).to_operator(), [ef.qpe_bits+ef.n])  # Postselect last qubit being |1>
    # Now trace out QPE and ancilla qubits
    return partial_trace(svec, list(range(ef.qpe_bits)) + [ef.qpe_bits + ef.n])


if __name__ == "__main__":
    # Specify error and matrix size
    error = 2**(-3)
    dim = 2**3

    # Choose random Hermitian matrix to run algorithm on by choosing random diagonal matrix and conjugating by unitary
    d = [0.19]+[0.7123 for _ in range(dim-1)]
    mat = np.diag(d)
    # un = unitary_group.rvs(dim)
    # mat = un @ np.diag(d) @ un.conj().T

    rho = get_ground_state(mat, error)
    probs = rho.probabilities()  # Get probabilities
    probs /= sum(probs)  # Normalize
    print(probs)

