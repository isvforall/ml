import numpy as np


def is_multiset(x, y):
    x.sort()
    y.sort()
    diff = x == y
    if False in diff:
        return False
    return True


def is_multiset_np(x, y):
    x.sort()
    y.sort()
    return np.all(x == y)
