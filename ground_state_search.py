from eig_search import *
import numpy as np
from random import random
from scipy.stats import unitary_group
import qiskit
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector, partial_trace


def get_ground_state(matrix, epsilon):
    # Step 1: Get estimate \theta_0
    # ef = EigenvalueFinding(matrix, epsilon/2)
    # theta0 = find_min(ef)
    theta0 = 0.101

    # Step 2: Construct the Grover circuit
    ef = EigenvalueFinding(matrix, epsilon/4)  # Need a bit mroe precision
    objective_qubits = list(range(ef.qpe_bits))

    oracle_list = [int(abs(x/2**ef.qpe_bits - theta0) < epsilon/2) for x in range(2**ef.qpe_bits)]
    good_states = [bin(x)[2:] for x in range(2**ef.qpe_bits) if abs(x/2**ef.qpe_bits - theta0) < epsilon/2]

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
    qc = transpile(qc, backend)
    job = backend.run(qc)
    svec = job.result().get_statevector(qc, decimals=7)

    # Step 4: Trace out the clock register
    rho = partial_trace(svec, list(range(ef.n, ef.n + ef.qpe_bits)))
    return rho


if __name__ == "__main__":
    # Specify error and matrix size
    error = 2**(-3)
    bits_of_precision = ceil(log2(1 / error))
    dim = 2**3

    # Choose random Hermitian matrix to run algorithm on by choosing random diagonal matrix and conjugating by unitary
    d = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.5]  # Can't have eigs too close to 0 or 1 due to QPE wraparound
    un = unitary_group.rvs(dim)
    mat = un @ np.diag(d) @ un.conj().T
    print(get_ground_state(mat, error))
