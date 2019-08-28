import numpy as np
from numba import jit

@jit(nopython=True)
def gillespie(model, x0, t_end):
    R = model.R
    buffer_size = 10**4

    times = np.empty(buffer_size)
    state_path = np.empty((buffer_size, len(x0)))
    path_index = 0
    num_jumps = np.zeros(R)
    
    x = x0
    t = 0.
            
    while True:
        intensities = model.get_intensities(x)
        total_intensity = np.sum(intensities)
        
        times[path_index] = t
        state_path[path_index,:] = x
        path_index += 1
        
        if path_index >= buffer_size:
            times = np.concatenate((times, np.empty(buffer_size)))
            empty = np.empty((buffer_size, len(x0)))
            state_path = np.vstack((state_path, empty))
        
        dt = np.random.exponential(1 / total_intensity)
        
        #check if simulation is over
        if t + dt > t_end:
            t = t_end
            times[path_index] = t
            state_path[path_index,:] = x
            path_index += 1
            break

        t += dt

        #get and execute the next reaction
        probabilities = intensities / total_intensity

        #due to numpy rounding errors,
        #it often thinks the probabilities sum to more than 1
        temp = np.random.multinomial(1, probabilities * 0.9999999999999)
        reaction_index = np.argmax(temp)
        num_jumps[reaction_index] += 1
        x = x + model.S[:, reaction_index]

    return times[0:path_index], state_path[0:path_index, :], num_jumps