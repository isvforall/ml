import numpy as np


def max_before_zero(x):
    max = 0
    prev = -1
    for n in x:
        if prev == 0 and n > max:
            max = n
        prev = n
    return max


def max_before_zero_np(x):
    zeros = x == 0
    return np.max(x[1:][zeros[:-1]])
