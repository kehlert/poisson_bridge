from poisson_bridge import PoissonBridge

import numpy as np

from scipy.stats import binom
from scipy.stats import poisson

class BinomialBridge(PoissonBridge):
    def __init__(self, T, discretization_power, points):
        assert 2**discretization_power == points.shape[0]
        self.T = T
        PoissonBridge.__init__(self, discretization_power, points)
        
    def generate(self):
        dim, n = self.points.shape

        Y = np.zeros((dim+1,n), dtype=int)
        Y[-1,:] = poisson.ppf(self.points[0,:], self.T)
        points_index = 1 #used the zeroth row of points already

        indices_step =  2 ** (self.discretization_power - 1)

        for i in range(0, self.discretization_power):
            indices = list(range(0, dim+1, indices_step))
            new_Y_indices = indices[1::2]
            prev_Y_indices = indices[0::2]

            point_indices = range(points_index, points_index + len(new_Y_indices))
            points_index += len(new_Y_indices)

            binomial_n = Y[prev_Y_indices[1:],:] - Y[prev_Y_indices[0:-1],:]

            #scipy raises a warning if we give binom.ppf a parameter of 0
            mask = (binomial_n > 0)
            points_mask = self.points[point_indices,:][mask]
            binomial_n_mask = binomial_n[mask]

            binomials = np.zeros(binomial_n.shape)
            binomials[mask] = binom.ppf(points_mask, binomial_n_mask, 0.5).astype(int)

            #scipy's binom.ppf cannot handle binomial(0, 0.5)
            #it returns nan, so we set the entries manually

            Y[new_Y_indices,:] = binomials + Y[prev_Y_indices[0:-1],:]
            indices_step //= 2

        self.times = np.linspace(0, self.T, dim+1)
        self.Y = Y
    
    def fill_jump_times(self):
        assert(self.times is not None)
        assert(self.Y is not None)
        
        n = self.Y.shape[1]
        times_diff = np.diff(self.times)

        #for each path
        for i in range(0, n):
            Y_vals = self.Y[:,i]
            unifs = np.random.uniform(size=Y_vals[-1])
            split_unifs = [np.sort(arr) for arr in np.split(unifs, Y_vals)]
            
            #first and last are always empty,
            #because they correspond to jumps before 0 and after T, respectively
            assert(len(split_unifs[0]) == 0)
            assert(len(split_unifs[-1]) == 0)
            split_unifs = split_unifs[1:-1]

            scaled_unifs = [t + dt * unifs for t,dt,unifs in zip(self.times[:-1], times_diff, split_unifs)]
            self.jump_times.append(np.concatenate(scaled_unifs))
