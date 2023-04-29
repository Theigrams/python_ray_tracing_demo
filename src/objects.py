import numpy as np

from src.lights import Ray
from src.materials import Material
from src.utils import normalize


class Primitive:
    def __init__(self, center: np.ndarray, material: Material):
        self.center = center
        self.material = material

    def intersect(self, ray: Ray):
        pass

    def get_normal(self, point: np.ndarray):
        pass


class Sphere(Primitive):
    def __init__(self, center, radius, material):
        super().__init__(center, material)
        self.radius = radius

    def intersect(self, ray):
        """calculate the intersection point of a ray and a sphere
        |CP| = |CO + OP| = |CO + t * d| = r^2
        <d,d> t^2 + 2<d,CO> t + <CO,CO> - r^2 = 0
        """
        co = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2 * co.dot(ray.direction)
        c = co.dot(co) - self.radius**2
        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None
        root = (-b - np.sqrt(discriminant)) / (2 * a)
        if root < 0.001 or root > 10e8:
            root = (-b + np.sqrt(discriminant)) / (2 * a)
            if root < 0.001 or root > 10e8:
                return None
        hit_point = ray.at(root)
        return hit_point

    def get_normal(self, point):
        return normalize(point - self.center)


class Plane(Sphere):
    def __init__(self, center, material, normal):
        """use a giant sphere to simulate the plane

        Args:
            normal : the normal of the plane
        """
        virtual_center = center - normal * 10000
        virtual_radius = 10000
        super().__init__(virtual_center, virtual_radius, material)
