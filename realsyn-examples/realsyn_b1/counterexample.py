'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-8.673382951280335, -5.103178418082997], \
        [-7.673382951280335, -7.103178418082997], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0])

    inputs = None
    end_point = [-0.10947145371949124, 0.15215728439468335]
    start_point = [0.5, 1.5]

    step = 0.1
    max_time = 0.8

    normal_vec = [1.0, 0.0]
    normal_val = -0.1

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
