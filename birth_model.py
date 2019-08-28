import numpy as np
from numba import jitclass
from numba import float64, int64

spec = [
    ('S', int64[:,:]),
    ('rate_constants', float64[:]),
    ('R', int64),
]

@jitclass(spec)
class BirthModel():
    def __init__(self):
        self.S = np.array([[1]], dtype=np.int64)
        self.rate_constants = np.array([1], dtype=np.float64)
        self.R = len(self.rate_constants)
          
    def get_intensities(self, x):
        temp = np.array([x[0]], dtype=np.float64)
        temp = np.maximum(temp, 0)
        return np.multiply(self.rate_constants, temp)
    