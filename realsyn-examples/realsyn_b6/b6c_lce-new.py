from hylaa.engine import HylaaSettings
from hylaa.containers import PlotSettings
from hylaa.hybrid_automaton import HyperRectangle, LinearConstraint
from controlcore.controller_eval import find_safe_controller, find_safe_controller_dce
from controlcore.lqr_safe_control import CntrlObject
import numpy as np

step_size = 0.01
max_time = 10


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 1
    plot_settings.ydim = 2

    s = HylaaSettings(step=step_size, max_time=max_time, disc_dyn=False, plot_settings=plot_settings)
    s.stop_when_error_reachable = False

    return s


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-2.1, -1.9), (-4.1, -3.9), (0, 0), (0, 0), (1.9, 2.1), (-4.1, -3.9), (0, 0), (0, 0), (0.0, 0.0)])
    usafe_arg = None
    if usafe_arg is None:
        usafe_constraint_list = [LinearConstraint([0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0], -1.4)]
        # usafe_constraint_list = [LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0], 1.6)]
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0], -3.0))

        usafe_arg = usafe_constraint_list

    a_matrix = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0],
                         [0, 0, -1.2, 0.1, 0, 0, 0, 0],
                         [0, 0, 0.1, -1.2, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0, -1.2, -0.1],
                         [0, 0, 0, 0, 0, 0, 0.1, -1.2]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    # (t_jumps, k_matrices) = find_safe_controller_dce(a_matrix, b_matrix, init_r, usafe_arg, settings)
    # print(t_jumps, k_matrices)
    cntrl_object = CntrlObject(a_matrix, b_matrix, init_r, usafe_arg, settings)
    cntrl_object.find_safe_controller(unit_wts_for_Q=False)
    print(cntrl_object.Q_matrices, cntrl_object.k_matrices)
