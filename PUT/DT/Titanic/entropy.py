from math import e, log
import numpy as np

def conditional_entropyXY(X, Y):
    conj_freq = {}
    prev_freq = {}
    for i in range(len(X)):
        if (X[i], Y[i]) in conj_freq:
            conj_freq[(X[i], Y[i])] += 1
        else:
            conj_freq[(X[i], Y[i])] = 1
        if X[i] in prev_freq:
            prev_freq[X[i]] += 1
        else:
            prev_freq[X[i]] = 1
    omega = sum(conj_freq.values())
    return -sum([conj_freq[(x, y)] / omega * log(conj_freq[(x, y)] / prev_freq[x], 2) for (x, y) in conj_freq])


def _entropy(y, base=None):
        #print("++++++++++++")
        #from icecream import ic
        #ic(y)
        value,counts = np.unique(y, return_counts=True)
        norm_counts = counts / counts.sum()
        base = e if base is None else base
        return -(norm_counts * np.log(norm_counts)/np.log(base)).sum()