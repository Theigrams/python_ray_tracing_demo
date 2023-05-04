import os
import sys

import numpy as np
import pytest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from assets.cornell_box import *
from src import *


def test_run():
    canvas1 = Canvas(100, 100)
    camera1 = Camera()
    sence1 = Sence(camera=camera1, objects=objects1, lights=lights1, canvas=canvas1)
    sence1.render(samples_per_pixel=1, max_depth=1)
    sence1.render_parallel(samples_per_pixel=1, max_depth=1, num_workers=2)
    canvas1.regular()


if __name__ == "__main__":
    pytest.main([__file__])
