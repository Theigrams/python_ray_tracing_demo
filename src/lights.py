import numpy as np

from src.utils import normalize


class Ray:
    """ray: p(t) = origin + t * direction"""

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + t * self.direction


class Light:
    def __init__(self, color, intensity=1.0):
        self.color = color
        self.intensity = intensity

    def get_irradiance(self, hit_point, normal):
        pass


class AmbientLight(Light):
    def __init__(self, color, intensity=1.0):
        super().__init__(color, intensity)

    def get_irradiance(self, hit_point, normal):
        return self.color * self.intensity


class PointLight(Light):
    def __init__(self, position, color, intensity=1.0):
        super().__init__(color, intensity)
        self.position = position

    def get_irradiance(self, hit_point, normal):
        return self.intensity / np.linalg.norm(hit_point - self.position) ** 2

    def get_direction(self, hit_point):
        return normalize(hit_point - self.position)


class DirectionalLight(Light):
    def __init__(self, direction, color, intensity=1.0):
        super().__init__(color, intensity)
        self.direction = direction

    def get_irradiance(self, hit_point, normal):
        return self.color * self.intensity

    def get_direction(self, hit_point):
        return self.direction
