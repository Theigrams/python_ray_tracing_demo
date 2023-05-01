import numpy as np


class Material:
    def __init__(self, ambient=1, diffuse=1, specular=1, color=np.array([1, 1, 1])):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.color = color


class WallMaterial(Material):
    def __init__(self, color=np.array([0.8, 0.8, 0.8])):
        super().__init__(ambient=1, diffuse=1, specular=0, color=color)


class FuzzyMaterial(Material):
    def __init__(self, color=np.array([0.8, 0.8, 0.8])):
        super().__init__(ambient=1, diffuse=0.5, specular=0.5, color=color)


class LightMaterial(Material):
    def __init__(self, color=np.array([10, 10, 10])):
        super().__init__(ambient=100, diffuse=100, specular=100, color=color)
