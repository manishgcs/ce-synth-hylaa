'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-2560.253332008155, -2594.566218119885, -4453.962172758664, -2559.999999626852, 0.0], \
        [0.25, 0.0, 0.0, 0.0, 0.0], \
        [0.0, 0.0019531, 0.0, 0.0, 0.0], \
        [0.0, 0.0, 0.0019531, 0.0, 0.0], \
        [0.0, 0.0, 0.0, 0.0, 0.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    inputs = None
    end_point = [4.11605282560395, 0.0, -1.5016049401325655, -1.5035181199244991, 1.1999999999999997]
    start_point = [-1.5, -1.4415340505447332, -1.5, -1.5, 0.4]

    step = 0.1
    max_time = 0.8

    normal_vec = [-1.0, 0.0, 0.0, 0.0, 0.0]
    normal_val = -3.8

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
