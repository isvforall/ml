# import numpy as np


def build_vec(x, v1, v2):
    result = []
    for i in range(len(v1)):
        result.append(x[v1[i]][v2[i]])
    return result


def build_vec_np(x, v1, v2):
    return x[v1, v2]
