import numpy as np



class Material:
    def __init__(self, ambient=1, diffuse=1, specular=1, color=np.array([1, 1, 1])):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.color = color
