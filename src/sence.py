import numpy as np

from src.materials import Material
from src.utils import normalize


class Sence:
    def __init__(self, camera, objects, lights):
        self.camera = camera
        self.objects = objects
        self.lights = lights

    def render(self):
        width, height = self.camera.resolution
        aspect_ratio = float(width) / height
        fov_adjustment = np.tan(self.camera.fov * np.pi / 360)
        image = np.zeros((height, width, 3))
        for x in range(width):
            for y in range(height):
                x_adjustment = (1 - (x + 0.5) / width) * aspect_ratio * fov_adjustment
                y_adjustment = (1 - (y + 0.5) / height) * fov_adjustment
                direction = normalize(
                    self.camera.direction
                    + x_adjustment * self.camera.right
                    + y_adjustment * self.camera.up
                )
                image[y][x] = self.ray_trace(self.camera.position, direction)
        return image
