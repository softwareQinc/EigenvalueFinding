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
    # ef = EigenvalueFinding(matrix, epsilon/2)
    # theta0 = find_min(ef)
    theta0 = 0.375  # Skip step 1 for testing purposes just to save time

    # Step 2: Construct the Grover circuit
    ef = EigenvalueFinding(matrix, epsilon/4)  # Need a bit more precision
    oracle_list = [0]*(2**(ef.qpe_bits + ef.n))
    good_states = []
    for x in range(2**ef.qpe_bits):
        if abs(x/2**ef.qpe_bits - theta0) < epsilon/2:
            for y in range(2**ef.n):
                z = bin(x)[2:].zfill(ef.qpe_bits) + bin(y)[2:].zfill(ef.n)
                good_states.append(z)
                oracle_list[int(z, 2)] = 1

    problem = AmplificationProblem(oracle=Statevector(oracle_list),
                                   state_preparation=algorithm_a(ef),
                                   is_good_state=good_states)
    grover = Grover(sampler=Sampler())

    # Step 3: Figure out how many iterations are needed
    result = grover.amplify(problem)
    n_iterations = result.iterations[-1]
    qc = grover.construct_circuit(problem=problem, power=n_iterations, measurement=False)
    # Step 4: Execute circuit and extract statevector
    backend = Aer.get_backend("statevector_simulator")
    # First we need to apply the bit oracle in order to postselect
    qc.add_register(QuantumRegister(1))
    qc.h(ef.qpe_bits + ef.n)

    po = QuantumCircuit(ef.qpe_bits + ef.n)
    po.diagonal([(-1)**(x in oracle_list) for x in range(2**(ef.qpe_bits + ef.n))], list(range(ef.qpe_bits + ef.n)))
    cpo = po.to_gate().control(num_ctrl_qubits=1)
    qc.append(cpo, [ef.qpe_bits + ef.n] + list(range(ef.qpe_bits + ef.n)))

    qc.h(ef.qpe_bits + ef.n)
    # qc.add_register(ClassicalRegister(1))
    # qc.measure(ef.qpe_bits + ef.n, 0)
    qc = transpile(qc, backend)
    job = backend.run(qc)
    print(job.result().get_statevector().probabilities([ef.qpe_bits+ef.n]))

    ancilla_result = False

    while not ancilla_result:
        job = backend.run(qc)
        ancilla_result = '1' in job.result().get_counts()

    svec = job.result().get_statevector(qc)  # Now postselect on |1>

    # Step 4: Trace out the clock register and the measured qubit
    return partial_trace(svec, list(range(ef.qpe_bits)) + [ef.qpe_bits+ef.n])


if __name__ == "__main__":
    # Specify error and matrix size
    error = 2**(-3)
    bits_of_precision = ceil(log2(1 / error))
    dim = 2**3

    # Choose random Hermitian matrix to run algorithm on by choosing random diagonal matrix and conjugating by unitary
    d = [0.375]+[0.875 for _ in range(7)]
    mat = np.diag(d)
    # un = unitary_group.rvs(dim)
    # mat = un @ np.diag(d) @ un.conj().T
    rho = get_ground_state(mat, error)
    print(rho.probabilities_dict())
