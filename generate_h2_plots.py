import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib

bond_length_step = 0.1
min_bond_length = 0.3
max_bond_length = 2.5001
bond_lengths = np.arange(min_bond_length, max_bond_length, bond_length_step)

with open("h2_data.txt", 'r') as f:
    lines = f.readlines()

approx = [float(line.split()[0]) for line in lines]
exact = [complex(line.split()[1]).real for line in lines]
error = [abs(approx[i] - exact[i]) for i in range(len(approx))]

plt.semilogy(bond_lengths, error, base=2)
plt.semilogy(bond_lengths, [0.01]*len(bond_lengths), base=2)
plt.xlabel("Bond length")
plt.ylabel("Error")
tikzplotlib.save("h2_error_plot.tex")
# plt.show()

# plt.plot(bond_lengths, approx)
# plt.plot(bond_lengths, exact)
# plt.xlabel("Bond length")
# plt.ylabel("Ground state energy")
# tikzplotlib.save("h2_energy_plot.tex")
# plt.show()
