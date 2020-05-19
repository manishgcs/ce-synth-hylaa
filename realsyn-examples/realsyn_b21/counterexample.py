'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-6.482371250480014, -8.334425154762455, -23.010672385900353, 0.0], \
        [2.0, 0.0, 0.0, 0.0], \
        [0.0, 0.5, 1.0, 0.0], \
        [0.0, 0.0, 0.0, 0.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 1.0])

    inputs = None
    end_point = [1.584831602886784, -7.931551242773427, 2.2635366085573976, 1.44]
    start_point = [1.0, 1.9, 1.9, 0.0]

    step = 0.01
    max_time = 1.44

    normal_vec = [0.0, 1.0, 0.0, 0.0]
    normal_val = -7.9

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
