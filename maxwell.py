import numpy as np
from ground_state_search import get_ground_state
from qiskit.quantum_info import partial_trace, Statevector


def get_discretization_matrix(N, d, epsilon_r):
    h = 1 / (0.5 + N)
    grid_points = [h / 2 + i * h for i in range(N)]
    second_derivative = np.diag([-2 / h ** 2] * N) + np.diag([1 / h ** 2] * (N - 1), 1) + np.diag(
        [1 / h ** 2] * (N - 1), -1)
    second_derivative[0][0] = -1 / h ** 2
    epsilon_matrix = np.diag([1 / (epsilon_r + 1) if x < d / 2 else 1 for x in grid_points])
    return epsilon_matrix @ second_derivative


if __name__ == "__main__":
    N = 2 ** 2
    d = 1.0
    epsilon_r = 4.0
    mat = get_discretization_matrix(N, d, epsilon_r)
    eigs = np.linalg.eigvals(mat)

    # print(mat)

    print(np.sort(np.abs(eigs)))
    print()

    scale_factor = 1.1 * np.max([abs(x) for x in eigs])
    zer = np.zeros((N, N))
    hermitian_matrix = np.block([[zer, mat], [mat.transpose(), zer]])

    # print(hermitian_matrix)
    print(np.sort(np.abs(np.linalg.eigvals(hermitian_matrix))))
    assert False

    hermitian_matrix += scale_factor * np.identity(2 * N)
    hermitian_matrix /= 2 * scale_factor

    rho = get_ground_state(hermitian_matrix, epsilon=(2 ** -1) / (2 * scale_factor))
    rho = partial_trace(rho, [0])  # First qubit comes from block matrix, so we don't need it

    rho.evolve(Statevector(psi_0)).trace()

    print()
