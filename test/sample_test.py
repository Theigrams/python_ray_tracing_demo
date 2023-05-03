import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import matplotlib.pyplot as plt

from src.utils import *


def test_random_in_unit_sphere():
    fuzz = 0.2
    v = np.array([1, 1, 0])
    N = 10000
    R = np.zeros((N, 3))
    for i in range(N):
        R[i] = normalize(v + fuzz * random_in_unit_sphere())
    T = np.arctan2(R[:, 1], R[:, 0]) / np.pi * 180
    plt.hist(T, bins=100)
    plt.show()


if __name__ == "__main__":
    pytest.main([__file__])
