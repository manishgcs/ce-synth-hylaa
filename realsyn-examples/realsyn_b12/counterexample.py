'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-0.9980723499451429, 0.0], \
        [0.0, 0.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 1.0])

    inputs = None
    end_point = [0.49999999999999994, 1.7999999999999998]
    start_point = [3.0143459440671965, 0.0]

    step = 0.01
    max_time = 1.8

    normal_vec = [-1.0, 0.0]
    normal_val = -0.4

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
