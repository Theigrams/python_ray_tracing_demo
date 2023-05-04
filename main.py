import time

import matplotlib.pyplot as plt
import numpy as np

from assets.cornell_box import *
from src import *

if __name__ == "__main__":
    canvas1 = Canvas(400, 400)
    camera1 = Camera()
    sence1 = Sence(camera=camera1, objects=objects1, lights=lights1, canvas=canvas1)
    t1 = time.time()
    # sence1.render(samples_per_pixel=4, max_depth=1)
    plt.ion()
    plt.figure(1)
    N = 100
    t = np.linspace(0, 20, 100)
    for i in tqdm(range(1, N + 1)):
        sence1.render_parallel(samples_per_pixel=4, max_depth=10, num_workers=7)
        plt.imshow(np.sqrt(np.clip(canvas1.pixels / i, 0, 1)))
        plt.pause(1)
        plt.clf()
    t2 = time.time()
    canvas1.pixels /= N
    canvas1.regular()
    print(f"Render time: {t2-t1}")
    canvas1.save("output/image.png")
