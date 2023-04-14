import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, QuantumRegister
from qiskit.circuit.library import Diagonal
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.primitives import Sampler
from qiskit.quantum_info import Statevector, partial_trace

# Step 1: Define the problem
# We would like to amplify states |110> and |111>
oracle = Statevector([0,0,0,0,0,0,1,1])  # Entry i is the output of a bit-query oracle when query is i.

# Specifying `state_preparation` to prepare a superposition of |01>, |10>, and |11>
# theta = 2 * np.arccos(1 / np.sqrt(3))
# state_preparation = QuantumCircuit(3)
# state_preparation.ry(theta, 0)
# state_preparation.ch(0,1)
# state_preparation.x(1)
# state_preparation.h(2)
n=6
state_preparation = QuantumCircuit(n)
oracle = Statevector([0]*(2**n - 1) + [1])  # Entry i is the output of a bit-query oracle when query is i.
state_preparation.h(range(n))
problem = AmplificationProblem(oracle=oracle, state_preparation=state_preparation, is_good_state=['1'*n])

# problem = AmplificationProblem(oracle=oracle, state_preparation=state_preparation, is_good_state=['110', '111'])
# End of step 1.

# Step 2: Run Grover to determine how many iterations are necessary
grover = Grover(sampler=Sampler())
result = grover.amplify(problem)
print(result.iterations, result.top_measurement)
iterations = result.iterations[-1]
# End step 2
#
# # Step 3: Run the Grover circuit using the just-calculated number of iterations
# qc = grover.construct_circuit(problem=problem, power=iterations, measurement=False)
# # Need to prepare a bit flip oracle. Basically we're doing phase kickback in reverse
# # Let O be a phase oracle (easy to prepare since it's diagonal). Then we append a qubit in the |+> state,
# # apply a controlled-O and then apply a Hadamard to the ancilla. This gives us a bit oracle.
# qc.add_register(QuantumRegister(1))
# qc.h(3)
# qc.diagonal([1]*8 + [1, 1, 1, 1, 1, 1, -1, -1], [0, 1, 2, 3])  # Controlled phase oracle
# qc.h(3)
# # We now need to postselect on the last (index=3) qubit being in state |1>.
# # Qiskit has no postselection tool, so we will run the entire circuit and then slice the statevector.
#
# backend = Aer.get_backend("statevector_simulator")
# qc = transpile(qc, backend)
# job = backend.run(qc)
# result = job.result()
# vec = result.get_statevector(qc)
# vec = np.asarray(vec)
# vec = vec[8:]  # Corresponds to postselecting 1
# print(vec)  # OMG IT WORKS!!!!
