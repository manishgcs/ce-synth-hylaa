'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [0.96065997, 0.1947354], \
        [-0.1947354, 0.96065997], \
        ])

    b_matrix = None
    c_vector = np.array([0.39340481, -0.03933961])

    inputs = None
    end_point = [0.36607819527734675, 1.0581049848184751]
    start_point = [-6.0, 0.0]

    step = 0.2
    max_time = 7.6000000000000005

    normal_vec = [1.0, 0.0]
    normal_val = 2.0

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
