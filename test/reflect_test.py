import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from src.utils import reflect


@pytest.mark.parametrize(
    "v,n,expected",
    [
        ([1, 0], [0, 1], [1, 0]),
        ([1, -1], [0, 1], [1, 1]),
        ([1, 0, 0], [0, 1, 0], [1, 0, 0]),
    ],
)
def test_reflect(v, n, expected):
    v = np.array(v) / np.linalg.norm(v)
    n = np.array(n) / np.linalg.norm(n)
    expected = np.array(expected) / np.linalg.norm(expected)
    result = reflect(v, n)
    assert np.allclose(result, expected)


if __name__ == "__main__":
    pytest.main([__file__])
