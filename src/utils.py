import os

import numpy as np
from PIL import Image

BLUE_COLOR = np.array([0, 0, 1])
RED_COLOR = np.array([1, 0, 0])
BLACK_COLOR = np.array([0, 0, 0])
WHITE_COLOR = np.array([1, 1, 1])


def normalize(v):
    return v / np.linalg.norm(v)


def reflect(v, n):
    return v - 2 * np.dot(v, n) * n


def refract(v, n, ref_idx):
    """calculate refracted vector

    Args:
        v : incident vector
        n : normal vector
        ref_idx : relative refractive index, n_out / n_in
    Returns:
        refracted vector
    """
    cos_theta = min(np.dot(-v, n), 1.0)
    r_out_perp = 1 / ref_idx * (v + cos_theta * n)
    r_out_parallel = -np.sqrt(abs(1.0 - np.linalg.norm(r_out_perp) ** 2)) * n
    r_out = r_out_perp + r_out_parallel
    return r_out


def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0**2
    return r0 + (1 - r0) * (1 - cosine) ** 5


def random_in_unit_sphere():
    while True:
        p = np.random.uniform(-1, 1, 3)
        if np.linalg.norm(p) < 1:
            return p


def random_unit_vector():
    return normalize(random_in_unit_sphere())


def animate(image_folder, save_name, fps=10):
    """animate images to a gif

    Args:
        image_folder : folder contains images
        filename : output filename
        fps : frames per second
    """

    filenames = []
    for filename in os.listdir(image_folder):
        if filename.endswith(".png") and filename != "image.png":
            filenames.append(filename)
    filenames.sort(key=lambda x: int(x.split(".")[0]))
    images = []
    for filename in filenames:
        try:
            images.append(Image.open(os.path.join(image_folder, filename)))
        except:
            pass
    Image.Image.save(
        images[0],
        save_name,
        save_all=True,
        append_images=images[1:],
        duration=1000 // fps,
        loop=0,
    )
