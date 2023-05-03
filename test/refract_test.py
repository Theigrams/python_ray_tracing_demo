import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from src.utils import refract


@pytest.mark.parametrize(
    "v,n,ref_idx",
    [
        ([1, 0], [0, 1], 1.5),
        ([1, -1], [0, 1], 1.5),
        ([1, 0, 0], [0, 1, 0], 1.5),
    ],
)
def test_refract(v, n, ref_idx):
    v = np.array(v) / np.linalg.norm(v)
    n = np.array(n) / np.linalg.norm(n)
    r_out = refract(v, n, ref_idx)

    assert np.allclose(np.linalg.norm(r_out), 1)

    sin_theta_in = np.linalg.norm(np.cross(v, n))
    sin_theta_out = np.linalg.norm(np.cross(r_out, n))
    assert np.allclose(sin_theta_in / sin_theta_out, ref_idx)


if __name__ == "__main__":
    pytest.main([__file__])
