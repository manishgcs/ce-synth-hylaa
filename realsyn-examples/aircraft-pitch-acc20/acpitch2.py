from hylaa.engine import HylaaSettings
from hylaa.containers import PlotSettings
from hylaa.hybrid_automaton import HyperRectangle, LinearConstraint
from controlcore.controller_eval import find_safe_controller, find_safe_controller_dce
from controlcore.lqr_safe_control import CntrlObject
import numpy as np

step_size = 0.01
max_time = 10

'''https://www.autonomousrobotslab.com/uploads/5/8/4/4/58449511/10_flightcontrol_lq.pdf'''


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 4
    plot_settings.ydim = 2

    s = HylaaSettings(step=step_size, max_time=max_time, disc_dyn=False, plot_settings=plot_settings)
    s.stop_when_error_reachable = False

    return s


# The deepest ce is '[1.         0.54862919 0.5        0.4        0.        ]' with depth '-0.41999999999999993'
# The deepest ce is '[1.  0.7 0.8 0.4 0. ]' with depth '-0.5214989022538624'
if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(1.0, 1.5), (0.5, 0.7), (0.5, 0.8), (0.3, 0.4), (0, 0)])
    # init_r = HyperRectangle([(1.0, 1.0), (0.54862919, 0.7), (0.5, 0.8), (0.4, 0.4), (0, 0)])
    usafe_arg = None

    if usafe_arg is None:
        usafe_constraint_list = [LinearConstraint([0.0, 0.0, 1.0, 0.0, 0.0], -0.42)]
        usafe_arg = usafe_constraint_list

    a_matrix = np.array([[-0.003, 0.039, 0, -0.332], [-0.065, -0.319, 7.74, 0], [0.020, -0.101, -0.429, 0], [0, 0, 1, 0]], dtype=float)

    b_matrix = np.array([[0.010], [-0.180], [-1.16], [0]], dtype=float)

    # (t_jumps, k_matrices) = find_safe_controller_dce(a_matrix, b_matrix, init_r, usafe_arg, settings)
    # print(t_jumps, k_matrices)
    cntrl_object = CntrlObject(a_matrix, b_matrix, init_r, usafe_arg, settings)
    cntrl_object.find_safe_controller(unit_wts_for_Q=False)
    print(cntrl_object.Q_matrices, cntrl_object.k_matrices)
