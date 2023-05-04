import multiprocessing as mp
from typing import List

import numpy as np
from tqdm import tqdm

from src.camera import *
from src.geometry import Primitive
from src.lights import Light
from src.materials import *
from src.utils import *


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
            color += ray_color(self, ray, max_depth)
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

    def hit(self, ray: Ray):
        hit_record = None
        closest_so_far = np.inf
        for obj in self.objects:
            hit_event = HitEvent(obj, ray)
            if hit_event.flag and hit_event.distance < closest_so_far:
                hit_record = hit_event
                closest_so_far = hit_event.distance
        return hit_record


def ray_color(sence, cam_ray, depth):
    # set Russian roulette probability p_RR = 0.8
    # if random() > p_RR, stop tracing
    # else, continue tracing and return color / p_RR
    p_RR = 0.8
    if depth <= 0 or np.random.random() > p_RR:
        return BLACK_COLOR
    hit_record = sence.hit(cam_ray)
    if hit_record is None:
        return BLACK_COLOR
    obj = hit_record.obj
    local_color = obj.material.color
    if isinstance(obj.material, LightMaterial):
        return local_color
    if isinstance(obj.material, DiffuseMaterial):
        scatter_dir = hit_record.normal + random_unit_vector()
        scatter_ray = Ray(hit_record.point, scatter_dir)
        if np.dot(scatter_dir, hit_record.normal) < 0:
            return BLACK_COLOR
        return local_color * ray_color(sence, scatter_ray, depth - 1) / p_RR
    elif isinstance(obj.material, MetalMaterial):
        reflect_dir = reflect(cam_ray.direction, hit_record.normal)
        scatter_dir = reflect_dir
        scatter_ray = Ray(hit_record.point, scatter_dir)
        return local_color * ray_color(sence, scatter_ray, depth - 1) / p_RR
    elif isinstance(obj.material, FuzzyMaterial):
        fuzz = 0.4
        reflect_dir = reflect(cam_ray.direction, hit_record.normal)
        scatter_dir = reflect_dir + fuzz * random_in_unit_sphere()
        if np.dot(scatter_dir, hit_record.normal) < 0:
            return BLACK_COLOR
        scatter_ray = Ray(hit_record.point, scatter_dir)
        return local_color * ray_color(sence, scatter_ray, depth - 1) / p_RR
    elif isinstance(obj.material, GlassMaterial):
        n_air, n_glass = 1.0, 1.5
        # relative refractive index, n_out / n_in
        ref_idx = n_glass / n_air
        normal = hit_record.normal
        if not hit_record.front_face:
            ref_idx = 1 / ref_idx
            normal = -1 * normal
        cos_theta = min(np.dot(-cam_ray.direction, normal), 1.0)
        sin_theta = np.sqrt(1 - cos_theta**2)
        # Check total internal reflection
        cannot_refract = 1 / ref_idx * sin_theta > 1.0
        reflect_ratio = reflectance(cos_theta, ref_idx)
        if cannot_refract or reflect_ratio > np.random.random():
            reflect_dir = reflect(cam_ray.direction, normal)
            scatter_dir = reflect_dir
            scatter_ray = Ray(hit_record.point, scatter_dir)
            return local_color * ray_color(sence, scatter_ray, depth - 1) / p_RR
        else:
            refract_dir = refract(cam_ray.direction, normal, ref_idx)
            scatter_dir = refract_dir
            scatter_ray = Ray(hit_record.point, scatter_dir)
            return local_color * ray_color(sence, scatter_ray, depth - 1) / p_RR
    else:
        return BLACK_COLOR


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
        self.normal = obj.get_normal(hit_point)
        self.front_face = bool(np.dot(self.normal, ray.direction) < 0)
