import time

import numpy as np

from assets.cornell_box import *
from src import *

if __name__ == "__main__":
    canvas1 = Canvas(600, 600)
    camera1 = Camera()
    sence1 = Sence(camera=camera1, objects=objects1, lights=lights1, canvas=canvas1)
    t1 = time.time()
    # sence1.render(samples_per_pixel=4, max_depth=1)
    sence1.render_parallel(samples_per_pixel=4, max_depth=1, num_workers=7)
    t2 = time.time()
    canvas1.regular()
    print(f"Render time: {t2-t1}")
    canvas1.save("output/image.png")
