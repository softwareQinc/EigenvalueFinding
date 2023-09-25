"""Purpose: Generate data for plot of error for 1000 random diagonal matrices"""

from eig_search import find_min, EigenvalueFinding
from random import random
from math import ceil, log2
import numpy as np

# Specify error and matrix size
error = 2**(-6)
bits_of_precision = ceil(log2(1 / error))
dim = 2**3

for _ in range(1000):
    print("Working on instance number ", _)
    # Choose random diagonal matrix
    d = [error + (1-2*error) * random() for _ in range(dim)]  # Can't have eigs close to 0 or 1 due to QPE wraparound
    lambda_0 = min(d)
    mat = np.diag(d)

    # Find the answer
    y_seq = find_min(EigenvalueFinding(mat, error))
    abs_error = [abs(lambda_0 - y_i) for y_i in y_seq]

    with open('eig_search_errors.txt', 'a') as f:
        for e in abs_error:
            f.write(str(e) + ' ')
        f.write('\n')
