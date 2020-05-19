'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-16.030577326559694, -154.3707920828445], \
        [7.4506e-09, 0.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0])

    inputs = None
    end_point = [19.2595287865929, -1.9999985744652145]
    start_point = [-1.0, -2.0]

    step = 0.1
    max_time = 10.0

    normal_vec = [-1.0, 0.0]
    normal_val = -0.2

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
