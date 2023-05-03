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
            color += ray_tracing(self, ray, max_depth)
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

    def add_shadow(self, hit_record):
        point = hit_record.point
        obj = hit_record.obj
        shadow_weight = 1.0
        for light in self.lights:
            light_dir = light.get_direction(point)
            light_ray = Ray(light.position, light_dir)
            hit_from_light = self.hit(light_ray)
        if hit_from_light.obj is not obj:
            if isinstance(hit_from_light.obj.material, GlassMaterial):
                shadow_weight = 0.5
            else:
                shadow_weight = 0
        return shadow_weight


def lambertian_reflect(lights, hit_record):
    hit_point = hit_record.point
    obj = hit_record.obj
    normal = obj.get_normal(hit_point)
    default_color = np.array([0.0, 0.0, 0.0])
    color = obj.material.color
    for light in lights:
        light_dir = light.get_direction(hit_point)
        # light_irradiance = light.get_irradiance(hit_point, normal)
        cos_theta = max(0, -np.dot(normal, light_dir))
        default_color += color * cos_theta
    default_color = np.clip(default_color, 0, 1)
    return default_color


def specular_reflect(lights, cam_ray, hit_record):
    hit_point = hit_record.point
    obj = hit_record.obj
    normal = obj.get_normal(hit_point)
    default_color = np.array([0.0, 0.0, 0.0])
    color = obj.material.color
    for light in lights:
        light_dir = light.get_direction(hit_point)
        H = normalize(light_dir + cam_ray.direction)
        cos_theta = max(0, -np.dot(normal, H))
        default_color += color * cos_theta**10
    default_color = np.clip(default_color, 0, 1)
    return default_color


def blinn_phong_shading(sence, cam_ray, hit_record):
    lights = sence.lights
    ambient_color = hit_record.obj.material.color
    ambient_weight = hit_record.obj.material.ambient
    diffuse_color = lambertian_reflect(lights, hit_record)
    diffuse_weight = hit_record.obj.material.diffuse
    specular_color = specular_reflect(lights, cam_ray, hit_record)
    specular_weight = hit_record.obj.material.specular
    reflection_color = diffuse_weight * diffuse_color + specular_weight * specular_color
    shadow_weight = sence.add_shadow(hit_record)
    local_color = reflection_color * shadow_weight + ambient_color * ambient_weight
    return local_color


def ray_tracing(sence, cam_ray, depth):
    if depth <= 0:
        return BLACK_COLOR
    hit_record = sence.hit(cam_ray)
    if hit_record is None:
        return BLACK_COLOR
    obj = hit_record.obj
    attenuation = obj.material.attenuation
    if isinstance(obj.material, LightMaterial):
        return obj.material.color
    local_color = blinn_phong_shading(sence, cam_ray, hit_record)
    if isinstance(obj.material, MetalMaterial):
        reflect_color = mirror_reflect(sence, cam_ray, hit_record, depth)
        return local_color * 0.1 + attenuation * reflect_color
    elif isinstance(obj.material, FuzzyMaterial):
        reflect_color = fuzz_metal_reflect(sence, cam_ray, hit_record, depth)
        return local_color * 0.5 + attenuation * reflect_color
    elif isinstance(obj.material, GlassMaterial):
        reflect_color = glass_reflect(sence, cam_ray, hit_record, depth)
        return local_color * 0.1 + attenuation * reflect_color
    else:
        return local_color


def mirror_reflect(sence, cam_ray, hit_record, depth):
    reflect_dir = reflect(cam_ray.direction, hit_record.normal)
    reflect_ray = Ray(hit_record.point, reflect_dir)
    return ray_tracing(sence, reflect_ray, depth - 1)


def fuzz_metal_reflect(sence, cam_ray, hit_record, depth):
    fuzz = 0.4
    reflect_dir = reflect(cam_ray.direction, hit_record.normal)
    # repeat N times to get a more smooth result
    cache_color = np.array([0.0, 0.0, 0.0])
    N = 20
    for _ in range(N):
        scatter_dir = normalize(reflect_dir + fuzz * random_in_unit_sphere())
        if np.dot(scatter_dir, hit_record.normal) < 0:
            continue
        reflect_ray = Ray(hit_record.point, scatter_dir)
        cache_color += ray_tracing(sence, reflect_ray, depth - 1)
    return cache_color / N


def glass_reflect(sence, cam_ray, hit_record, depth):
    if depth <= 0:
        return np.zeros(3)
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
    reflect_dir = reflect(cam_ray.direction, normal)
    reflect_ray = Ray(hit_record.point, reflect_dir)
    reflect_color = ray_tracing(sence, reflect_ray, depth - 1)
    if cannot_refract:
        return reflect_color
    refract_dir = refract(cam_ray.direction, normal, ref_idx)
    refract_ray = Ray(hit_record.point, refract_dir)
    refract_color = ray_tracing(sence, refract_ray, depth - 1)
    return reflect_color * reflect_ratio + refract_color * (1 - reflect_ratio)


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
