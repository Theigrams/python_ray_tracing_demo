import matplotlib.pyplot as plt
import numpy as np

from src.lights import Ray
from src.utils import normalize


class Camera:
    def __init__(
        self,
        look_from=np.array([0.0, 1.0, 5.0]),
        look_at=np.array([0.0, 1.0, 1.0]),
        fov=60,
        aspect_ratio=1,
    ):
        # the position of the camera
        self.look_from = look_from
        # the point the camera is looking at
        self.look_at = look_at

        self.Fwd = normalize(self.look_at - self.look_from)
        self.Right = normalize(np.cross(self.Fwd, np.array([0, 1, 0])))
        self.Up = normalize(np.cross(self.Right, self.Fwd))

        self.aspect_ratio = aspect_ratio
        # use vertical field-of-view (fovY)
        self.fov = fov
        self.viewport = ViewPort(self)


class ViewPort:
    def __init__(self, camera: Camera):
        theta = np.deg2rad(camera.fov)
        # the default distance between the camera and the viewport is 1
        half_height = np.tan(theta / 2)
        half_width = camera.aspect_ratio * half_height
        self.cam_coord = camera.look_from
        self.height = 2 * half_height
        self.width = 2 * half_width

        # the horizontal and vertical vectors of the viewport
        self.horizontal = self.width * camera.Right
        self.vertical = self.height * camera.Up

        # the lower left corner of the viewport in camera basis
        self.lower_left = (
            camera.look_from + camera.Fwd - self.horizontal / 2 - self.vertical / 2
        )

    def get_ray(self, u, v):
        """get the ray from the camera to the viewport in camera basis (u,v)
        note that u,v is in [0,1]
        """
        pixel_coord = self.lower_left + u * self.horizontal + v * self.vertical
        direction = normalize(pixel_coord - self.cam_coord)
        return Ray(self.cam_coord, direction)


class Canvas:
    def __init__(self, width=200, height=200):
        self.resolution = (width, height)
        self.width = width
        self.height = height
        self.pixels = np.zeros((width, height, 3))

    def set_pixel(self, i, j, color):
        self.pixels[self.height - j - 1, i] += color

    def save(self, filename):
        plt.imsave(filename, np.sqrt(self.pixels))

    def regular(self):
        self.pixels = np.clip(self.pixels, 0, 1)
