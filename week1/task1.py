import numpy as np


def prod_nonzero_ondiagonal(array):
    i, j = 0, 0
    n = array.size
    m = array[0].size
    prod = 1
    while i != n and j != m:
        e = array[i][j]
        if e != 0:
            prod *= e
        i += 1
        j += 1
    return prod


def prod_nonzero_ondiagonal_np(array):
    d = array.diagonal()
    return np.prod(d[d > 0])
