from src import *

light_source = Sphere(
    center=np.array([0, 5.4, 1]),
    radius=3.0,
    material=LightMaterial(),
)

ground = Plane(
    center=np.array([0, -0.5, 1]),
    material=WallMaterial(),
    normal=np.array([0, 1, 0]),
)

back_wall = Plane(
    center=np.array([0, 1, -1]),
    material=WallMaterial(),
    normal=np.array([0, 0, 1]),
)

right_wall = Plane(
    center=np.array([1.5, 0, 1]),
    material=WallMaterial(color=np.array([0.6, 0.0, 0.0])),
    normal=np.array([-1, 0, 0]),
)

left_wall = Plane(
    center=np.array([-1.5, 0, 1]),
    material=WallMaterial(color=np.array([0.0, 0.6, 0.0])),
    normal=np.array([1, 0, 0]),
)


ceiling = Plane(
    center=np.array([0, 2.5, 1]),
    material=WallMaterial(),
    normal=np.array([0, -1, 0]),
)

diffuse_ball = Sphere(
    center=np.array([0, -0.2, 1.5]),
    radius=0.3,
    material=Material(color=np.array([0.8, 0.3, 0.3])),
)
metal_ball = Sphere(
    center=np.array([0.8, 0.2, 1]),
    radius=0.7,
    material=Material(color=np.array([0.6, 0.8, 0.8])),
)
glass_ball = Sphere(
    center=np.array([-0.7, 0.0, 0.5]),
    radius=0.5,
    material=Material(color=np.array([1.0, 1.0, 1.0])),
)
fuzz_metal_ball = Sphere(
    center=np.array([-0.6, -0.3, 2.0]),
    radius=0.2,
    material=FuzzyMaterial(color=np.array([0.8, 0.6, 0.2])),
)

light1 = PointLight(
    position=np.array([0, 5.4 - 3.1, 1]), color=light_source.material.color
)

objects1 = [
    light_source,
    ground,
    back_wall,
    right_wall,
    left_wall,
    ceiling,
    diffuse_ball,
    metal_ball,
    glass_ball,
    fuzz_metal_ball,
]

lights1 = [light1]
