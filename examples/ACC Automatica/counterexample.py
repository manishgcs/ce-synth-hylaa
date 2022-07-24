'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [0, -1, 1, 0, 0], \
        [0, 0, 0, 1, 0], \
        [0, 0, 0, 0, 0], \
        [1, -4, 3, -3, 0], \
        [0, 0, 0, 0, 0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, -10.0, 1.0])

    inputs = None
    end_point = [15.050908142453716, 15.0, 20.0, 1.7663930340922434, 3.0]
    start_point = [2.0, 21.753389566320514, 20.0, -1.0, 0.0]

    step = 1
    max_time = 3

    normal_vec = [0.0, 1.0, 0.0, 0.0, 0.0]
    normal_val = 15.0

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
