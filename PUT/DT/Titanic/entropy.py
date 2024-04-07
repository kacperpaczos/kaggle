from math import e, log
import numpy as np
from icecream import ic

def conditional_entropyXY(X, Y):
    ic(X)
    ic(Y)
    conj_freq = {}
    prev_freq = {}
    for (x_index, x_val), (y_index, y_val) in zip(X.items(), Y.items()):
        if (x_val, y_val) in conj_freq:
            conj_freq[(x_val, y_val)] += 1
        else:
            conj_freq[(x_val, y_val)] = 1
        if x_val in prev_freq:
            prev_freq[x_val] += 1
        else:
            prev_freq[x_val] = 1
    omega = sum(conj_freq.values())
    return -sum([conj_freq[(x, y)] / omega * log(conj_freq[(x, y)] / prev_freq[x], 2) for (x, y) in conj_freq])


def entropy(y, base=None):
        #print("++++++++++++")
        #from icecream import ic
        #ic(y)
        value,counts = np.unique(y, return_counts=True)
        norm_counts = counts / counts.sum()
        base = e if base is None else base
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()