from poisson_bridge import PoissonBridge

import numpy as np
from scipy.stats import beta
from scipy.stats import gamma

class BetaBridge(PoissonBridge):
    def __init__(self, Y_at_T, discretization_power, points):
        assert 2**discretization_power == points.shape[0]
        self.Y_at_T = Y_at_T
        PoissonBridge.__init__(self, discretization_power, points)
        
    def generate(self):
        dim, n = self.points.shape
        
        Y = np.floor(np.linspace(0, self.Y_at_T, dim+1)).astype(int)

        times = np.zeros((dim+1,n))
        times[-1,:] = gamma.ppf(self.points[0,:], Y[-1])
        points_index = 1 #used the zeroth row of points already
        indices_step =  2 ** (self.discretization_power - 1)

        for i in range(0, self.discretization_power):
                indices = list(range(0, dim+1, indices_step))
                new_times_indices = indices[1::2]
                prev_times_indices = indices[0::2]

                point_indices = range(points_index, points_index + len(new_times_indices))
                points_index += len(new_times_indices)

                alpha_params = Y[new_times_indices] - Y[prev_times_indices[0:-1]]
                beta_params = Y[prev_times_indices[1:]] - Y[prev_times_indices[0:-1]] - alpha_params

                alpha_params = np.tile(alpha_params, (n, 1)).T
                beta_params = np.tile(beta_params, (n, 1)).T

                betas = beta.ppf(self.points[point_indices,:], alpha_params, beta_params)

                scale =  times[prev_times_indices[1:]] - times[prev_times_indices[0:-1]]
                times[new_times_indices,:] = scale * betas + times[prev_times_indices[0:-1]]

                indices_step //= 2
                
        self.times = times
        self.Y = Y
        
    def fill_jump_times(self):
        assert(self.times is not None)
        assert(self.Y is not None)
        
        n = self.times.shape[1]

        num_known_times = len(self.Y) - 1
        num_unknown_jump_times = self.Y[-1] - num_known_times
        unifs = np.random.uniform(size=(n, num_unknown_jump_times))
        
        split_unifs = [np.sort(arr, axis=1) for arr in np.split(unifs, self.Y[1:]-np.arange(1,len(self.Y)), axis=1)]
        assert(split_unifs[-1].size == 0)
        split_unifs = split_unifs[:-1]
        
        times_diff = np.diff(self.times, axis=0)
        scaled_unifs = [self.times[i,:] + np.multiply(times_diff[i,:], unifs.T) for i, unifs in enumerate(split_unifs)]
 
        new_jump_times = np.vstack(scaled_unifs)
        jump_times = np.sort(np.vstack((new_jump_times, self.times)), axis=0).T

        #need to put self.jump_times in the same format as the binomial jump times
        self.jump_times = [jump_times[i,1:] for i in range(0, n)]
