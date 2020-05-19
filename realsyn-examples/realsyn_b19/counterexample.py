'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [5.1950117647058823e-17, -3.3414588235294116e-17, 5.879341176470588e-32, 0.0], \
        [2.7756e-17, 0.0, 0.0, 0.0], \
        [0.0, 2.7756e-17, 1.0, 0.0], \
        [0.0, 0.0, 0.0, 1.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 0.1])

    inputs = None
    end_point = [0.0, 0.0, 0.0, 0.30000000000000004]
    start_point = [0.0, 0.0, 0.0, 0.2]

    step = 0.1
    max_time = 0.1

    normal_vec = [1.0, 0.0, -1.0, 0.0]
    normal_val = 0.5

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
