'Counter-example trace generated using HyLAA'

import sys
import numpy as np
from hylaa.check_trace import check, plot

def check_instance():
    'define parameters for one instance and call checking function'

    # dynamics x' = Ax + Bu + c
    a_matrix = np.array([\
        [1.0, 0.0, 0.1, 0.0, 0.0], \
        [0.0, 1.0, 0.0, 0.1, 0.0], \
        [-221.3271161412532, -0.015681415800135746, -12.043764810929071, -0.0007856714017244991, 0.0], \
        [-0.015681415808892363, -221.32711614125986, -0.0007856714017244991, -12.043764810929042, 0.0], \
        [0.0, 0.0, 0.0, 0.0, 0.0], \
        ])

    b_matrix = None
    c_vector = np.array([0.0, 0.0, 0.0, 0.0, 1.0])

    inputs = None
    end_point = [-2.112084066805579, -3.8592309590732325, 42.2551937230812, 75.41848759196112, 0.3400000000000001]
    start_point = [-2.5, -4.5, 0.0, 0.0, 0.33000000000000007]

    step = 0.01
    max_time = 0.01

    normal_vec = [0.0, 0.0, 0.0, -1.0, 0.0]
    normal_val = -75.0

    sim_states, sim_times = check(a_matrix, b_matrix, c_vector, step, max_time, start_point, inputs, end_point)

    if len(sys.argv) < 2 or sys.argv[1] != "noplot":
        plot(sim_states, sim_times, inputs, normal_vec, normal_val, max_time, step)

if __name__ == "__main__":
    check_instance()
