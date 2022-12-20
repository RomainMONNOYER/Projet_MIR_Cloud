import math

import cv2
import numpy as np

def euclidean(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    return np.linalg.norm(l1 - l2)


def chiSquareDistance(l1, l2):
    s = 0.0
    for i, j in zip(l1, l2):
        if i == j == 0.0:
            continue
        s += (i - j) ** 2 / (i + j)
    return s


def bhatta(l1, l2):
    l1 = np.array(l1)
    l2 = np.array(l2)
    num = np.sum(np.sqrt(np.multiply(l1, l2, dtype=np.float64)), dtype=np.float64)
    den = np.sqrt(np.sum(l1, dtype=np.float64) * np.sum(l2, dtype=np.float64))
    return math.sqrt(1 - num / den)


def flann(a, b):
    a = np.float32(np.array(a))
    b = np.float32(np.array(b))
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.inf
    index_params = dict(algorithm=1, trees=5)
    sch_params = dict(checks=50)
    flannMatcher = cv2.FlannBasedMatcher(index_params, sch_params)
    matches = list(map(lambda x: x.distance, flannMatcher.match(a, b)))
    return np.mean(matches)


def bruteForceMatching(a, b):
    a = np.array(a).astype('uint8')
    b = np.array(b).astype('uint8')
    if a.shape[0] == 0 or b.shape[0] == 0:
        return np.inf
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = list(map(lambda x: x.distance, bf.match(a, b)))
    return np.mean(matches)






