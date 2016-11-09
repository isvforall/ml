import numpy as np


def run_length(array):
    nums = np.array([], dtype='int')
    times = np.array([], dtype='int')
    prev = array[0]
    count = 0
    for e in array:
        if prev == e:
            count += 1
        else:
            nums = np.append(nums, prev)
            times = np.append(times, count)
            count = 1
        prev = e
    nums = np.append(nums, prev)
    times = np.append(times, count)
    return (nums, times)


def run_length_np(array):
    c = np.bincount(array)
    return (np.unique(array), c[c > 0])
