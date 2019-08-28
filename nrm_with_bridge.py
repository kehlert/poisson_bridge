import numpy as np
from numba import jit

@jit(nopython=True)
def nrm_with_bridge(model, x0, t_end, internal_jump_times):
    R = model.R
    buffer_size = 10**4

    times = np.empty(buffer_size)
    state_path = np.empty((buffer_size, len(x0)))
    path_index = 0

    internal_next_jump_times = np.zeros(R)
    internal_times = np.zeros(R)
    num_jumps = np.zeros(R)
    
    x = x0
    t = 0.
    
    for r in range(0, R):
        if num_jumps[r] < len(internal_jump_times[r]):
            internal_next_jump_times[r] = internal_jump_times[r][int(num_jumps[r])]
        else :
            internal_next_jump_times[r] = np.random.exponential()
            
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
        
        time_to_next_jumps = (internal_next_jump_times - internal_times) / intensities
        reaction_index = np.argmin(time_to_next_jumps)
        dt = time_to_next_jumps[reaction_index]
        
#         print('t: {}'.format(t))
#         print('x: {}'.format(x))
#         print()
        
#         print('P: {}'.format(internal_next_jump_times))
#         print('T: {}'.format(internal_times))
#         print('num jumps: {}'.format(num_jumps))
#         print('internal jump times:')
#         for r in range(0, R):
#             print('r: {}'.format(r))
#             print(internal_jump_times[r][0:int(num_jumps[r]+1)])
#             print()
#         print()
        
#         print('intensities: {}'.format(intensities))
#         print('time to next: {}'.format(time_to_next_jumps))
#         print('reaction index, dt: {}, {}'.format(reaction_index, dt))
#         print()
#         print('---------------------\n')
        
        #check if simulation is over
        if t + dt > t_end:
            internal_times += (t_end - t) * intensities
            t = t_end
            times[path_index] = t
            state_path[path_index,:] = x
            path_index += 1
            break

        t += dt
        internal_times += dt * intensities

        num_jumps[reaction_index] += 1
        
        if num_jumps[reaction_index] < len(internal_jump_times[reaction_index]):
            internal_next_jump_times[reaction_index] = internal_jump_times[reaction_index][int(num_jumps[reaction_index])]
        else:
            internal_next_jump_times[reaction_index] = internal_times[reaction_index] + np.random.exponential()

        x = x + model.S[:, reaction_index]

    return times[0:path_index], state_path[0:path_index, :], num_jumps