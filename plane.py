class Plane:
    def __init__(self, points):
        self.x, self.y, self.z, self.d = points

    def __repr__(self):
        return f"({self.x}, {self.y}, {self.z}), d={self.d}"
