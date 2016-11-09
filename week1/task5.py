import numpy as np


def make_gray(image):
    channels = np.array([0.299, 0.587, 0.114])
    res = []
    for i in range(len(image)):
        row = []
        for j in range(len(image[i])):
            r = image[i][j][:3] * channels
            row.append(int(sum(r)))
        res.append(row)
    return res


def make_gray_np(image):
    # channels = np.array([0.299, 0.587, 0.114])
    return np.dot(image[..., :3], [0.299, 0.587, 0.114]).astype('int')
