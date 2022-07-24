
from hylaa.engine import HylaaSettings
from hylaa.containers import PlotSettings
from hylaa.hybrid_automaton import HyperRectangle, LinearConstraint
from controlcore.controller_eval import find_safe_controller, find_safe_controller_dce
import numpy as np

step_size = 0.01
max_time = 10


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []
    rv.append((ha.modes['loc0'], init_r))

    return rv


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
    init_r = HyperRectangle([(0.9, 1.0), (0.9, 1.9), (0.9, 1.9), (0.0, 0.0)])

    usafe_arg = None
    if usafe_arg is None:
        # usafe_constraint_list = [LinearConstraint([1.0, 0.0, 0.0, 0.0], -10)]
        usafe_constraint_list = [LinearConstraint([-1.0, 0.0, 0.0, 0.0], -1.5)]
        # usafe_constraint_list = [LinearConstraint([0.0, 1.0, 0.0, 0.0], -6.1)]
        usafe_arg = usafe_constraint_list

    a_matrix = np.array([[2.9589, -1.4589, 0.95887], [2, 0, 0], [0, 0.5, 1]], dtype=float)
    b_matrix = np.array([[2], [0], [0]], dtype=float)

    (t_jumps, k_matrices) = find_safe_controller_dce(a_matrix, b_matrix, init_r, usafe_arg, settings)
    print(t_jumps, k_matrices)

    settings = define_settings()
    init_r = HyperRectangle([(0.9, 1.0), (0.9, 1.9), (0.9, 1.9), (0.0, 0.0)])
    usafe_r = None

    # a_matrix_ext, b_matrix_ext = extend_a_b(a_matrix, b_matrix)
    #
    # Q1 = np.array([[1, 0, 0],
    #                [0, 1, 0],
    #                [0, 0, 1]], dtype=float)
    #
    # u_dim = len(b_matrix[0])
    # R1 = 0.01 * np.eye(u_dim)
    # k_matrix1 = get_input(a_matrix, b_matrix, Q1, R1)
    #
    # a_bk_matrix1 = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix1)
    # pv_object = run_hylaa(settings, init_r, [a_bk_matrix1], [int(10 / step_size)], usafe_r)
    # lce_object = pv_object.compute_longest_ce(control_synth=True)
    #
    # t_jumps = lce_object.switching_times
    # k_matrices = [k_matrix1]
    #
    # Q2 = np.array([[4.51, 0, 0],
    #                [0, 0.72, 0],
    #                [0, 0, 0.73]], dtype=float)
    #
    # R2 = 0.01 * np.eye(u_dim)
    # k_matrix2 = get_input(a_matrix, b_matrix, Q2, R2)
    # k_matrices.append(k_matrix2)
    #
    # Q3 = np.array([[1, 0, 0],
    #                [0, 1, 0],
    #                [0, 0, 1]], dtype=float)
    #
    # R3 = 0.01 * np.eye(u_dim)
    # k_matrix3 = get_input(a_matrix, b_matrix, Q3, R3)
    # k_matrices.append(k_matrix3)
    #
    # a_bk_matrices = []
    # for idx in range(len(k_matrices)):
    #     k_matrix = k_matrices[idx]
    #     a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
    #     a_bk_matrices.append(a_bk_matrix)
    #
    # pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_r)
    # depth_direction = np.identity(len(init_r.dims))
    # for idx in range(len(init_r.dims)-1):
    #     deepest_ce_1 = pv_object.compute_deepest_ce(depth_direction[idx])
    #     # print(deepest_ce_1.ce_depth)
    #     deepest_ce_2 = pv_object.compute_deepest_ce(-depth_direction[idx])
    #     print("depth difference: {}".format(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth)))
    #     # print(deepest_ce_2.ce_depth)
