import numpy as np

DAFAULT_COLOR = np.array([0.8, 0.8, 0.8])


class Material:
    def __init__(
        self,
        ambient=0.05,
        diffuse=1,
        specular=1,
        color=DAFAULT_COLOR,
        attenuation=0.5,
    ):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.color = color
        self.attenuation = attenuation


class DiffuseMaterial(Material):
    def __init__(self, color=DAFAULT_COLOR):
        super().__init__(color=color, diffuse=1, specular=0)
        self.attenuation = self.color


class MetalMaterial(Material):
    def __init__(self, color=DAFAULT_COLOR):
        super().__init__(color=color, attenuation=0.5)


class FuzzyMaterial(Material):
    def __init__(self, color=DAFAULT_COLOR):
        super().__init__(color=color, diffuse=0.5, specular=0.5, attenuation=0.5)


class GlassMaterial(Material):
    def __init__(self, color=DAFAULT_COLOR):
        super().__init__(color=color, diffuse=0.2, specular=0.8, attenuation=0.95)


class LightMaterial(Material):
    def __init__(self, color=np.array([10, 10, 10])):
        super().__init__(ambient=100, diffuse=100, specular=100, color=color)
