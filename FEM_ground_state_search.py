from math import sqrt
import matplotlib as plt
from eig_search import *
from ground_state_search import get_ground_state
import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector


def fem_matrix(N, d, epsilon_r, x_0=0.):
    h = 2 / (N + 1)  # Interval is [-1, 1], so solve for h in -1 + (N+1)h = 1
    grid_points = np.linspace(-1, 1, num=N + 2)
    epsilon = [0 if abs(x + h / 2 - x_0) > d / 2 else epsilon_r for x in grid_points]  # size = N+2
    matrix = np.diag([(-1 / h) * (1 / (1 + epsilon[i]) + 1 / (1 + epsilon[(i + 1)])) for i in range(N)]) + \
             np.diag([(1 / h) * (1 / (1 + e)) for e in epsilon[1:-2]], +1) + \
             np.diag([(1 / h) * (1 / (1 + e)) for e in epsilon[1:-2]], -1)

    return matrix


if __name__ == "__main__":
    error = 2 ** (-2)
    dim = 2 ** 4

    mat = fem_matrix(dim, 10 / 17, 5., x_0=-6 / 17)

    ev0, p = np.linalg.eigh(mat)
    psi_0 = p[:, 0]

    # Here, we cheat a little and use a priori knowledge of what lambda_0 is, so that we can use fewer clock qubits
    shift_val = abs(min(min(ev0), 0)) + error / 2 + max(abs(ev0)) / 5
    scale_factor = max(abs(ev0 + shift_val)) * (1 + error)

    mat = (mat + shift_val * np.identity(dim)) / scale_factor
    ev = (ev0 + shift_val) / scale_factor
    ev = np.sort(ev)

    # error = 2**np.floor(log2(error/scale_factor))
    delta_energy = ev[1] - ev[0]

    error = min(error, 2 ** np.floor(log2(delta_energy)))
    print("Number of qubits in the clock register:", int(-log2(error)))

    lambda0, rho = get_ground_state(mat, error)
    overlap = rho.expectation_value(Statevector(psi_0).to_operator())

    # rho.__array__()

    # export the matrices
    rho_exact = np.outer(psi_0, psi_0.conjugate())
    pd.DataFrame(np.round(np.real(rho), 4)).to_csv("data/re_num_rho.csv", header=None, index=None)
    pd.DataFrame(np.round(np.imag(rho), 4)).to_csv("data/im_num_rho.csv", header=None, index=None)
    pd.DataFrame(np.round(np.real(rho_exact), 4)).to_csv("data/re_exact_rho.csv", header=None, index=None)
    pd.DataFrame(np.round(np.imag(rho_exact), 4)).to_csv("data/im_exact_rho.csv", header=None, index=None)

    print("Eigs are", ev)
    print("The estimated/real values for lambda are (bare)", [lambda0, min(abs(ev))])
    print("The estimated/real values for lambda are (rescaled)", [lambda0*scale_factor-shift_val, min(abs(ev))*scale_factor-shift_val])
    print("The error is (bare):", error)
    print("The error is (rescaled):", error*scale_factor)
    print("Overlap is", overlap.real)  # Ignore small imaginary component coming from roundoff error
