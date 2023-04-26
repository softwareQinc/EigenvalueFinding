import numpy as np
from math import log2, ceil
from random import random
from scipy.linalg import expm
from qiskit import QuantumCircuit, Aer
from qiskit.extensions import UnitaryGate
from qiskit.algorithms import PhaseEstimation, IterativeAmplitudeEstimation, EstimationProblem
from qiskit.utils import QuantumInstance


class EigenvalueFinding:
    """Class containing all information necessary for solving the eigenvalue problem. Some parameters are redundant but
     are still handy to carry around."""
    def __init__(self, matrix, epsilon):
        self.mat = matrix
        self.epsilon = epsilon
        self.m = ceil(log2(1 / epsilon))
        self.N = matrix.shape[0]
        self.n = int(log2(self.N))
        self.delta = 1 / (2 * self.N + 2)
        self.qpe_bits = self.m + ceil(log2(2 + 1 / (2 * self.delta)))  # For simplicity, we don't use the median trick
        self.q = 0.75*(1-self.delta)/self.N  #  Because we use iterative amplitude estimation, we have a different
                                             # threshold than in the paper


def bin2dec(b):
    """Purpose: Convert binary to decimal.
    Input: b is either a list of chars or a string
    Output: The corresponding decimal number, where the most significant bit is considered as the rightmost one
    Examples: bin2dec('100')==0.125 and bin2dec('01')==0.5."""
    return sum([0.5 ** i for i in range(1, len(b) + 1) if b[-i] == '1'])


def less_than(y):
    """Purpose: Prepare function that can be used by Amplitude Estimation to determine whether string is good or not
    Input: y
    Output: Indicator function for whether or not bin2dec(x) < y"""
    return lambda x: bin2dec(x) < y

def u0_circuit(n, matrix):
    """Purpose: Create circuit that prepares a uniform superposition of the eigenstates of a Hermitian/unitary matrix
    Input: Number n of qubits and the matrix
    Output: QuantumCircuit
    Note: In reality, we won't know how to create this circuit, and we will need to prepare \sum_j |j>|j>, however here
     we cheat in order to use fewer qubits in our simulation."""
    qc = QuantumCircuit(n)
    _, p = np.linalg.eigh(matrix)
    qc.h(range(n))
    qc.append(UnitaryGate(p), list(range(n)))
    return qc


def algorithm_a(eigfinding):
    """Purpose: Create quantum circuit corresponding to the algorithm \mathcal{A}
    Input: eigfinding object
    Output: QuantumCircuit encoding \mathcal{A}"""
    qpe = PhaseEstimation(num_evaluation_qubits=eigfinding.qpe_bits, quantum_instance=Aer.get_backend("aer_simulator"))
    qpe_circuit = qpe.construct_circuit(unitary=UnitaryGate(expm(2 * np.pi * 1j * eigfinding.mat)),
                                        state_preparation=u0_circuit(eigfinding.n, eigfinding.mat))
    return qpe_circuit


def amp_est(eigfinding, y, alpha=None):
    """Purpose: Run amplitude estimation on the algorithm \mathcal{A}
    Input: eigfinding object, float y for the function \chi_y, failure probability alpha.
    Output: Estimate of \Pr[X < y]."""
    if alpha is None:
        alpha = 1 - 0.99 ** (1 / eigfinding.m)  # This ensures the overall success probability is >= 0.99

    problem = EstimationProblem(state_preparation=algorithm_a(eigfinding=eigfinding),
                                objective_qubits=list(range(eigfinding.qpe_bits)),
                                is_good_state=less_than(y))
    backend = Aer.get_backend("aer_simulator")
    quantum_instance = QuantumInstance(backend)
    epsilon_ae = 0.25 * (1-eigfinding.delta)/eigfinding.N  # Required precision
    ae = IterativeAmplitudeEstimation(epsilon_target=epsilon_ae, alpha=alpha, quantum_instance=quantum_instance)
    return ae.estimate(problem).estimation


def find_min(eigfinding):
    """Purpose: Iterate the QAE subroutine to compute y_0, y_1, ...
    Input: eigenfinding object
    Output: List of estimates [y_0, y_1, ..., y_m]"""
    y = 0.5  # y_0
    y_seq = [y]
    for i in range(1, eigfinding.m + 1):
        ptilde = amp_est(eigfinding=eigfinding, y=y)
        if ptilde > eigfinding.q:
            y -= 1/2**(i+1)
        else:
            y += 1/2**(i+1)
        y_seq.append(y)
        # print("y_{0} = {1}".format(i, y))
    return y_seq


if __name__ == "__main__":  # Run algorithm on random matrix
    error=2**-7
    dim=8
    d = [0.1 + 0.8*random() for _ in range(dim)]
    mat = np.diag(d)
    ef = EigenvalueFinding(mat,error)
    find_min(ef)

    # Specify error and matrix size
    # error = 2**(-7)
    # bits_of_precision = ceil(log2(1 / error))
    # dim = 2**3
    #
    # for _ in range(10):
    #     print("Working on instance number ", _)
    #     # Choose random Hermitian matrix to run algorithm on by choosing random diagonal matrix and conjugating by unitary
    #     d = [0.1 + 0.8 * random() for _ in range(dim)]  # Can't have eigs too close to 0 or 1 due to QPE wraparound
    #     lambda_0 = min(d)
    #     un = unitary_group.rvs(dim)
    #     mat = un @ np.diag(d) @ un.conj().T
    #     # eigvals = np.sort(np.linalg.eigvalsh(mat))
    #     # print("eigenvalues of M are {0}".format(eigvals))
    #
    #     # Find the answer
    #     ef = EigenvalueFinding(mat, error)
    #     y_seq = find_min(ef)
    #     abs_error = [abs(lambda_0 - y_i) for y_i in y_seq]
    #
    #     with open('eig_search_errors.txt', 'a') as f:
    #         for e in abs_error:
    #             f.write(str(e) + ' ')
    #         f.write('\n')
    #     # estimated_minimum = find_min(ef)[-1]
    #
    # # Print data
    # # true_error = abs(min(d) - estimated_minimum)
    # # print("\nlambda_0 = {0}".format(min(d)))
    # # print("Algorithm estimates lambda_0 ~ {0}".format(estimated_minimum))
    # # print("true_error / desired_error = {0} (should be <1)".format(true_error / error))




