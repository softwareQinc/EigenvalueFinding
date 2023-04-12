import numpy as np
import qiskit
from qiskit import QuantumCircuit, Aer, transpile
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector

oracle = Statevector([0,0,0,0,0,0,1,1])

theta = 2 * np.arccos(1 / np.sqrt(3))
state_preparation = QuantumCircuit(3)
state_preparation.ry(theta, 0)
state_preparation.ch(0,1)
state_preparation.x(1)
state_preparation.h(2)

problem = AmplificationProblem(oracle=oracle, state_preparation=state_preparation, is_good_state=['110', '111'])

grover = Grover(sampler=Sampler())
# qc = grover.construct_circuit()
result = grover.amplify(problem)
iterations = result.iterations[-1]
qc = grover.construct_circuit(problem=problem, power=iterations, measurement=False)

backend = Aer.get_backend("statevector_simulator")
qc = transpile(qc, backend)
job = backend.run(qc)
result = job.result()
print(result.get_statevector(qc, decimals=3))
