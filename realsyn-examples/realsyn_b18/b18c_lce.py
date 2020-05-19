
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt
from control import *

step_size = 0.1


def define_ha(settings, args):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    a_matrices = args[0]
    t_jumps = list(args[1])
    usafe_r = args[2]

    n_locations = len(a_matrices)
    n_variables = len(a_matrices[0][0])
    print(n_locations, n_variables)

    ha.variables = []
    for idx in range(n_variables-1):
        x_var_name = "x"+str(idx)
        ha.variables.append(x_var_name)
    ha.variables.append("t")

    locations = []
    for idx in range(n_locations):
        loc_name = 'loc' + str(idx)
        loc = ha.new_mode(loc_name)
        loc.a_matrix = a_matrices[idx]
        c_vector = [0.0] * n_variables
        c_vector[n_variables-1] = 1
        loc.c_vector = c_vector
        if idx == 0:
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = 1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, step_size * t_jumps[idx]))
        elif idx == n_locations - 1:
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = -1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, -step_size * t_jumps[idx-1]))
        else:
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = -1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, -step_size * t_jumps[idx-1]))
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = 1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, step_size * t_jumps[idx]))
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans_vec = [0.0] * n_variables
        trans_vec[n_variables - 1] = -1.0
        trans.condition_list.append(LinearConstraint(trans_vec, -step_size*(t_jumps[idx]+1)))

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0], -3.8))
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0, 0.0, 0.0, 0.0], -0))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])

        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for idx in range(n_locations):
        trans_u = ha.new_transition(locations[idx], error)
        for constraint in usafe_set_constraint_list:
            trans_u.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


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

    s = HylaaSettings(step=step_size, max_time=100.0, disc_dyn=False, plot_settings=plot_settings)
    s.stop_when_error_reachable = False
    
    return s


def run_hylaa(settings, init_r, *args):

    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, args)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


def extend_a_b(A, B):
    A = A.copy().tolist()
    n_dim = len(A)
    A_ext = []
    for idx in range(n_dim):
        A_row = A[idx]
        A_row.append(0.0)
        A_ext.append(A_row)
    A_ext.append(np.zeros(n_dim+1, dtype=float))
    A_ext = np.array(A_ext)
    B_ext = B.copy().tolist()
    B_ext.append(np.zeros(len(B_ext[0]), dtype=float))
    B_ext = np.array(B_ext)
    return A_ext, B_ext


def get_input(A, B, Q, R):
    (X1, L, G) = care(A, B, Q, R)
    G = np.array(G)
    k_matrix = []
    for idx in range(len(G)):
        k_token = G[idx].copy().tolist()
        k_token.append(0.0)
        k_matrix.append(k_token)
    k_matrix = np.array(k_matrix, dtype=float)
    return k_matrix


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-1.5, -0.5), (-1.5, -0.5), (-1.5, -0.5), (-1.5, -0.5), (0, 0)])
    usafe_r = None

    a_matrix = np.array([[0.02366, -0.31922, 0.0012041, -4.0292e-17],
                         [0.25, 0, 0, 0],
                         [0, 0.0019531, 0, 0],
                         [0, 0, 0.0019531, 0]], dtype=float)

    b_matrix = np.array([[256], [0], [0], [0]], dtype=float)

    a_matrix_ext, b_matrix_ext = extend_a_b(a_matrix, b_matrix)

    Q1 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=float)

    u_dim = len(b_matrix[0])
    R1 = 0.01 * np.eye(u_dim)
    k_matrix1 = get_input(a_matrix, b_matrix, Q1, R1)

    a_bk_matrix1 = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix1)
    pv_object = run_hylaa(settings, init_r, [a_bk_matrix1], [int(100 / step_size)], usafe_r)
    lce_object = pv_object.compute_longest_ce(lpi_required=False, control_synth=True)

    t_jumps = lce_object.switching_times
    k_matrices = [k_matrix1]

    Q2 = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=float)

    R2 = 0.01 * np.eye(u_dim)
    k_matrix2 = get_input(a_matrix, b_matrix, Q2, R2)
    k_matrices.append(k_matrix2)

    Q3 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=float)

    R3 = 0.01 * np.eye(u_dim)
    k_matrix3 = get_input(a_matrix, b_matrix, Q3, R3)
    k_matrices.append(k_matrix3)

    Q4 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]], dtype=float)

    R4 = 0.01 * np.eye(u_dim)
    k_matrix4 = get_input(a_matrix, b_matrix, Q4, R4)
    k_matrices.append(k_matrix4)

    a_bk_matrices = []
    for idx in range(len(k_matrices)):
        k_matrix = k_matrices[idx]
        a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
        a_bk_matrices.append(a_bk_matrix)

    pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_r)
    lce_object = pv_object.compute_longest_ce(lpi_required=False, control_synth=True)
    depth_direction = np.identity(len(init_r.dims))
    for idx in range(len(init_r.dims)-1):
        deepest_ce_1 = pv_object.compute_deepest_ce(depth_direction[idx])
        # print(deepest_ce_1.ce_depth)
        deepest_ce_2 = pv_object.compute_deepest_ce(-depth_direction[idx])
        print("depth difference: {}".format(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth)))
        # print(deepest_ce_2.ce_depth)
