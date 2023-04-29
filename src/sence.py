import multiprocessing as mp
from typing import List

import numpy as np
from tqdm import tqdm

from src.camera import *
from src.lights import Light
from src.materials import Material
from src.objects import Primitive
from src.utils import normalize


class Sence:
    def __init__(
        self,
        camera: Camera,
        objects: List[Primitive],
        lights: List[Light],
        canvas: Canvas,
    ):
        self.camera = camera
        self.objects = objects
        self.lights = lights
        self.viewport = camera.viewport
        self.canvas = canvas

    def render(self, samples_per_pixel=4, max_depth=5):
        width, height = self.canvas.resolution
        for i in tqdm(range(width), desc="Rendering"):
            for j in range(height):
                _, color = self.render_pixel(i, j, samples_per_pixel, max_depth)
                self.canvas.set_pixel(i, j, color / samples_per_pixel)

    def render_pixel(self, i, j, samples_per_pixel=4, max_depth=5):
        color = np.zeros(3)
        for _ in range(samples_per_pixel):
            u = (i + np.random.random()) / self.canvas.width
            v = (j + np.random.random()) / self.canvas.height
            ray = self.viewport.get_ray(u, v)
            color += self.ray_color(ray, max_depth)
        return (i, j), color

    def render_parallel(self, samples_per_pixel=4, max_depth=5, num_workers=4):
        width, height = self.canvas.resolution
        with mp.Pool(processes=num_workers) as pool:
            # Map the render function to the list of pixels in the image
            results = pool.starmap(
                self.render_pixel,
                [
                    (i, j, samples_per_pixel, max_depth)
                    for i in range(width)
                    for j in range(height)
                ],
            )
            # Set the pixel color on the canvas
            for (i, j), color in results:
                self.canvas.set_pixel(i, j, color / samples_per_pixel)

    def ray_color(self, ray: Ray, depth):
        if depth <= 0:
            return np.zeros(3)
        # if we hit nothing, return background color
        default_color = np.array([1.0, 1.0, 1.0])
        hit_record = self.hit(ray)
        if hit_record is not None:
            default_color = hit_record.obj.material.color
        return default_color

    def hit(self, ray: Ray):
        hit_record = None
        closest_so_far = np.inf
        for obj in self.objects:
            hit_event = HitEvent(obj, ray)
            if hit_event.flag and hit_event.distance < closest_so_far:
                hit_record = hit_event
                closest_so_far = hit_event.distance
        return hit_record


class HitEvent:
    def __init__(self, obj, ray):
        hit_point = obj.hit(ray)
        if hit_point is None:
            self.flag = False
            return
        self.flag = True
        self.point = hit_point
        self.obj = obj
        self.distance = np.linalg.norm(hit_point - ray.origin)
