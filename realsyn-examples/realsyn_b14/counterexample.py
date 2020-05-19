'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [-48.250914153711165, -88.11823924497908, -279.9793116452655, 0.0], \
        [2.0, 0.0, 0.0, 0.0], \
        [0.0, 0.5, 1.0, 0.0], \
        [0.0, 0.0, 0.0, 0.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 1.0])

    inputs = None
    end_point = [-10.191665959604975, -2.202013660910655, 2.3384774674620603, 0.20000000000000007]
    start_point = [0.9, 1.9, 1.9, 0.05]

    step = 0.01
    max_time = 0.15

    normal_vec = [1.0, 0.0, 0.0, 0.0]
    normal_val = -10.0

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
