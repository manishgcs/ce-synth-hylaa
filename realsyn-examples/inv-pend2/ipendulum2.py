from hylaa.engine import HylaaSettings
from hylaa.containers import PlotSettings
from hylaa.hybrid_automaton import HyperRectangle, LinearConstraint
from controlcore.controller_eval import find_safe_controller, find_safe_controller_dce
from controlcore.lqr_safe_control import CntrlObject
import numpy as np

step_size = 0.01
max_time = 5

'''Optimal Control of Nonlinear Inverted Pendulum System Using PID Controller and LQR: 
Performance Analysis Without and With Disturbance Input'''
'''    Lal Bahadur Prasad, Barjeev Tyagi & Hari Om Gupta '''


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    s = HylaaSettings(step=step_size, max_time=max_time, disc_dyn=False, plot_settings=plot_settings)
    s.stop_when_error_reachable = False

    return s


if __name__ == '__main__':
    settings = define_settings()
    # init_r = HyperRectangle([(0.5, 1.0), (0.5, 1.0), (0, 0.5), (0, 0.5), (0, 0)])
    init_r = HyperRectangle([(-0.11, 0.11), (-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1), (0, 0)])
    usafe_arg = None

    if usafe_arg is None:
        # usafe_constraint_list = [LinearConstraint([0.0, -1.0, 0.0, 0.0, 0.0], -0.25)]  # R = 1. Can change x2 >= 0.3
        # usafe_constraint_list = [LinearConstraint([0.0, -1.0, 0.0, 0.0, 0.0], -0.4)]  # R = 0.1. Can change x2 >= 0.35
        usafe_constraint_list = [LinearConstraint([0.0, -1.0, 0.0, 0.0, 0.0], -0.45)]  # R = 0.01
        usafe_arg = usafe_constraint_list

    a_matrix = np.array([[0, 1, 0, 0], [29.8615, 0, 0, 0], [0, 0, 0, 1], [-0.9401, 0, 0, 0]],
                        dtype=float)

    b_matrix = np.array([[0], [-1.1574], [0], [0.4167]], dtype=float)

    # (t_jumps, k_matrices) = find_safe_controller_dce(a_matrix, b_matrix, init_r, usafe_arg, settings)
    # print(t_jumps, k_matrices)
    cntrl_object = CntrlObject(a_matrix, b_matrix, init_r, usafe_arg, settings)
    cntrl_object.find_safe_controller(unit_wts_for_Q=True)
    print(cntrl_object.Q_matrices, cntrl_object.k_matrices)
