import numpy as np
from random import random
from a_eig_search_cheating import find_min, EigenvalueFinding, amp_est
from math import log2, ceil, sqrt, pi
from time import time


def find_min_steps(eigfinding):
    # This is Algorithm 1 from Overleaf
    y = 0.5  # y_0

    tmp_y = []

    for i in range(1, eigfinding.m + 2):
        ptilde = amp_est(eigfinding=eigfinding, y=y)
        if ptilde > eigfinding.q:
            y -= 2 ** (-i)
        else:
            y += 2 ** (-i)
        #print("y_{0} = {1}".format(i, y))
        tmp_y.append(y)
    return tmp_y


if __name__ == "__main__":
    bitts = 6
    error = 0.5**bitts
    bits_of_precision = ceil(log2(1 / error))
    dim = 2**3

    all_points = []

    Ntrials = 1000
    for k in range(Ntrials):
        # Now we choose a random matrix to run the algorithm on
        eigvals = [error + (1-2*error) * random() for _ in range(dim)]
        mat = np.diag(eigvals)
        #print("M = {0}".format(mat))
        #print("eigenvalues of M are {0}".format(eigvals))
        #

        # Find the answer
        ef = EigenvalueFinding(mat, error)

        # Print some stuff about accuracy
        start = time()
        all_steps = find_min_steps(ef)
        end = time()
        estimated_minimum = all_steps[-1]
        diff_tmp = np.abs(np.array(all_steps)-min(eigvals))

        all_points.append(diff_tmp)

        #print("Time elapsed:", end-start)
        #print("\nlambda_0 = {0}".format(min(eigvals)))
        #print("Algorithm estimates lambda_0 ~ {0}".format(estimated_minimum))
        #print("Algorithm all rel_diff |lambda0-y_i| ~ {0}".format(diff_tmp))

        print("% line nr. ",k+1)
        print("\\addplot [ultra thin, lightredbkg]")
        print("table {%")
        for i,el in enumerate(diff_tmp):
            print(i,el)
        print("};")
        print()

    print()
    #print(all_points)
    avgs = [np.sum([el[i] for el in all_points])/Ntrials for i in range(bitts+1)]
    #print(avgs)

    print("% line with avg")
    print("\\addplot [very thick, powerredmn, dashed]")
    print("table {%")
    for i, el in enumerate(avgs):
        print(i, el)
    print("};")
    print()









