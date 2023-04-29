class Ray:
    """ray: p(t) = origin + t * direction"""

    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction

    def at(self, t):
        return self.origin + t * self.direction


class Light:
    def __init__(self, position, color):
        self.position = position
        self.color = color
