'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-326.9221330429018, -29.152537560606703, 289.61792082675913, -0.017236721081578463, 0.0], \
        [0.0, 1.0, 0.0, 0.1, 0.0], \
        [-327.9221330429018, -29.152537560606703, 290.4049208267591, -0.008336721081578463, 0.0], \
        [-1.9025707686533722, -239.10349133759644, 1.8942340475717936, -12.596368604543185, 0.0], \
        [0.0, 0.0, 0.0, 0.0, 0.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    inputs = None
    end_point = [-5.510287888041185, 1.3870271764205684, -6.064828739538344, -28.15283100235093, 0.38000000000000006]
    start_point = [0.7, 1.7, 0.0, 0.0, 0.11000000000000006]

    step = 0.01
    max_time = 0.27

    normal_vec = [1.0, 0.0, 0.0, 0.0, 0.0]
    normal_val = -5.5

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
