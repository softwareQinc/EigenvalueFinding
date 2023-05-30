"""Generates plot containing errors when running algorithm on random diagonal matrices, as well as the max and average error lines"""
import matplotlib.pyplot as plt
import tikzplotlib

sums = [0]*7
maxes = [0]*7
epsilon = 2**-6

plt.semilogy(range(7), [epsilon/2 + 2**-(i+1) for i in range(7)], 'k', linewidth=2, base=2)

with open("eig_search_errors.txt", 'r') as f:
    lines = f.readlines()

n_trials = len(lines)

for i in range(n_trials):
    xpts = [float(x) for x in lines[i].split()]
    for j in range(7):
        sums[j] += xpts[j]
        maxes[j] = max(maxes[j], xpts[j])
    if 980 <= i < 1000:
        if i == 0:
            plt.semilogy(xpts, 'blue', linewidth=0.4, base=2)
        else:
            plt.semilogy(xpts, 'blue', linewidth=0.4, base=2)

averages = [s/n_trials for s in sums]
plt.semilogy(averages, 'blue', linewidth=2, base=2)
plt.semilogy(maxes, 'green', linewidth=2, base=2)

plt.xlabel('$i$')
plt.ylabel('$|y_i - \lambda_0|$')
# plt.show()
tikzplotlib.save('new_error_plot.tex')
