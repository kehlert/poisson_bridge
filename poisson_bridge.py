class PoissonBridge:
    def __init__(self, discretization_power, points):
        self.discretization_power = discretization_power
        self.points = points
        self.times = None
        self.Y = None
        self.jump_times = []
