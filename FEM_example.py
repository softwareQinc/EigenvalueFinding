import numpy as np

np.set_printoptions(linewidth=np.inf)
from ground_state_search import get_ground_state
from qiskit.quantum_info import partial_trace, Statevector
import matplotlib.pyplot as plt


def get_matrix(N, d, epsilon_r, x_0=0):
    h = 2 / (N + 1)  # Interval is [-1, 1], so solve for h in -1 + (N+1)h = 1
    grid_points = np.linspace(-1, 1, num=N + 2)
    epsilon = [0 if abs(x + h / 2 - x_0) > d / 2 else epsilon_r for x in grid_points]  # size = N+2
    matrix = np.diag([(-1 / h) * (1 / (1 + epsilon[i]) + 1 / (1 + epsilon[(i + 1)])) for i in range(N)]) + \
             np.diag([(1 / h) * (1 / (1 + e)) for e in epsilon[1:-2]], +1) + \
             np.diag([(1 / h) * (1 / (1 + e)) for e in epsilon[1:-2]], -1)

    return matrix


if __name__ == "__main__":
    print(get_matrix(8, 2 / 3, -0.5, x_0=2 / 9))
