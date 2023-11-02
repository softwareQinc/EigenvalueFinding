"""Generates data for plotting effectiveness of median trick"""

import numpy as np
from eig_search import bin2dec
from qiskit import QuantumCircuit, Aer, ClassicalRegister, transpile
from qiskit.circuit.library import PhaseGate
from qiskit.algorithms import PhaseEstimation
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
from statistics import median
from itertools import product
import tikzplotlib

m = 6
epsilon = 2 ** -6
eig = 0.375 + 2 ** -8
state_preparation = QuantumCircuit(1)
state_preparation.x(0)
u = QuantumCircuit(1)
u.append(PhaseGate(2 * np.pi * eig), [0])

qpe = PhaseEstimation(num_evaluation_qubits=m, quantum_instance=Aer.get_backend("statevector_simulator"))
qpe_circuit = qpe.construct_circuit(state_preparation=state_preparation, unitary=u)
qpe_circuit.add_register(ClassicalRegister(1))
qpe_circuit.measure(6, 0)
backend = Aer.get_backend("statevector_simulator")
qpe_circuit = transpile(qpe_circuit, backend)
output_state = backend.run(qpe_circuit).result().get_statevector()
probs = Statevector(output_state).probabilities_dict()
probs = {s[1:]: probs[s] for s in probs if s[0] == '1'}
probs = {bin2dec("0b" + s): probs[s] for s in probs}
data = []
for x in range(2 ** 6):
    if x / 2 ** 6 in probs:
        data.append(probs[x / 2 ** 6])
    else:
        data.append(0)

median_data_5 = {x / 2 ** 6: 0 for x in range(2 ** 6)}
median_data_3 = {x / 2 ** 6: 0 for x in range(2 ** 6)}
for (x1, x2, x3, x4, x5) in product(range(2 ** 6), range(2 ** 6), range(2 ** 6), range(2 ** 6), range(2 ** 6)):
    median_data_5[median([x1, x2, x3, x4, x5]) / 2 ** 6] += data[x1] * data[x2] * data[x3] * data[x4] * data[x5]
for (x1, x2, x3) in product(range(2 ** 6), range(2 ** 6), range(2 ** 6)):
    median_data_3[median([x1, x2, x3]) / 2 ** 6] += data[x1] * data[x2] * data[x3]

plt.semilogy([x / 2 ** 6 for x in range(2 ** 6)], median_data_5.values(), label='Median trick with $c = 5$', base=2)
plt.semilogy([x / 2 ** 6 for x in range(2 ** 6)], median_data_3.values(), label='Median trick with $c = 3$', base=2)
plt.semilogy([x / 2 ** 6 for x in range(2 ** 6)], data, label='Without median trick', base=2)
plt.xlabel("$X$")
plt.ylabel("$\Pr[X]$")
plt.title("QPE probability distribution with and without the median trick ($\lambda = {0}$)".format(eig))
plt.legend()

# Uncomment to write data
# with open("median_plot_new.txt", 'a') as f:
#     for x in data:
#         f.write(str(x) + ' ')
#     f.write("\n")
#     for x in median_data_3.values():
#         f.write(str(x) + ' ')
#     f.write("\n")
#     for x in median_data_5.values():
#         f.write(str(x) + ' ')

plt.show()
