
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

step_size = 0.01
max_time = 10


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
    # ha.variables = ["x1", "x2", "x3", "x4", "t"]

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
            loc.inv_list.append(LinearConstraint(loc_inv_vec, -step_size * t_jumps[idx - 1]))
        else:
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = -1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, -step_size * t_jumps[idx - 1]))
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
        trans_vec = np.zeros(n_variables, dtype=float)
        trans_vec[0] = 1.0
        # usafe_set_constraint_list.append(LinearConstraint(trans_vec, -5.0))
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], -5.5))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0, 0.0, 0.0, 0], -0.5))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0, 0.0, 0.0, 0], 1.0))
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

    s = HylaaSettings(step=step_size, max_time=max_time, disc_dyn=False, plot_settings=plot_settings)
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


def find_safe_controller(a_matrix, b_matrix, init_r, usafe_r):
    a_matrix_ext, b_matrix_ext = extend_a_b(a_matrix, b_matrix)

    R_mult_factor = 0.01

    Q_matrix = np.eye(len(a_matrix[0]), dtype=float)

    u_dim = len(b_matrix[0])
    R_matrix = R_mult_factor * np.eye(u_dim)
    k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)

    a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
    old_pv_object = run_hylaa(settings, init_r, [a_bk_matrix], [int(max_time / step_size)], usafe_r)
    lce_object = old_pv_object.compute_longest_ce(control_synth=True)

    if lce_object.getCounterexample() is None:
        print("The system is safe")
        return None, k_matrix
    else:
        last_pv_object = None
        depth_direction = np.identity(len(init_r.dims))
        old_depth_diffs = []
        for idx in range(len(init_r.dims) - 1):
            deepest_ce_1 = old_pv_object.compute_deepest_ce(depth_direction[idx])
            deepest_ce_2 = old_pv_object.compute_deepest_ce(-depth_direction[idx])
            old_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

        t_jumps = lce_object.switching_times
        n_locations = len(t_jumps) + 1
        loc_idx = 1

        Q_mult_factors = np.empty(n_locations)
        Q_matrix = np.eye(len(a_matrix[0]))
        Q_matrices = [Q_matrix]
        k_matrices = [k_matrix]

        for idx in range(len(t_jumps)):
            Q_matrix = np.eye(len(a_matrix[0]))
            Q_matrices.append(Q_matrix)

            R_matrix = R_mult_factor * np.eye(u_dim)
            k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
            k_matrices.append(k_matrix)

        a_bk_matrices = []
        for idx in range(len(k_matrices)):
            k_matrix = k_matrices[idx]
            a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
            a_bk_matrices.append(a_bk_matrix)

        Q_matrix = np.multiply(Q_matrices[loc_idx], old_depth_diffs)
        k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
        a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[1])

        new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_r)
        depth_direction = np.identity(len(init_r.dims))
        new_depth_diffs = []
        for idx in range(len(init_r.dims) - 1):
            deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
            deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
            new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

        Q_weigths = []
        for idx in range(len(new_depth_diffs)):
            if new_depth_diffs[idx] - old_depth_diffs[idx] < 0:
                Q_weigths.append(old_depth_diffs[idx])
            else:
                Q_weigths.append(1/old_depth_diffs[idx])

        mult_factors = [10, 0.1]
        for m_factor in mult_factors:
            Q_mult_factor = m_factor
            Q_matrix = m_factor * np.multiply(Q_matrices[1], Q_weigths)
            k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
            a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[1])

            new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_r)
            u_interval = new_pv_object.compute_usafe_interval(loc_idx)
            if u_interval[0] == u_interval[1] == -1:
                last_pv_object = new_pv_object
                Q_mult_factors[loc_idx] = Q_mult_factor
                loc_idx = loc_idx + 1
                break
            depth_direction = np.identity(len(init_r.dims))
            new_depth_diffs = []
            for idx in range(len(init_r.dims) - 1):
                deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
                deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
                new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

            if new_depth_diffs < old_depth_diffs:
                Q_mult_factors.fill(1)
                Q_mult_factors[loc_idx] = Q_mult_factor
                break

        # Check once if the system is safe, after applying the controller on loc1
        if last_pv_object is not None:
            lce_object = last_pv_object.compute_longest_ce()
            if lce_object.getCounterexample() is None:
                print("The system is safe")
                print(Q_mult_factors, Q_weigths)
                return t_jumps, k_matrices

        Q_mult_factor = Q_mult_factors[1]
        assert Q_mult_factor == 0.1 or Q_mult_factor == 10
        for loc in range(loc_idx, n_locations):
            if last_pv_object is not None:
                u_interval = last_pv_object.compute_usafe_interval(loc)
                if u_interval[0] == u_interval[1] == -1:
                    continue
            else:
                print("Last pv object is None")
            current_loc_safe = False
            while current_loc_safe is False:
                Q_matrix = Q_matrices[loc]
                Q_mult_factors[loc] = Q_mult_factor * Q_mult_factors[loc]
                Q_matrix = Q_mult_factors[loc] * np.multiply(Q_matrix, Q_weigths)
                R_matrix = R_mult_factor * np.eye(u_dim)
                k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
                k_matrices[loc] = k_matrix
                a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
                a_bk_matrices[loc] = a_bk_matrix
                new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_r)
                u_interval = new_pv_object.compute_usafe_interval(loc_idx)
                if u_interval[0] == u_interval[1] == -1:
                    current_loc_safe = True
                    loc_idx = loc_idx + 1
                    last_pv_object = new_pv_object
        print(Q_mult_factors, Q_weigths)
        return t_jumps, k_matrices


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(0.3, 0.7), (1.3, 1.7), (0, 0), (0, 0), (0, 0)])
    usafe_r = None

    a_matrix = np.array([[1, 0, 0.1, 0],
                         [0, 1, 0, 0.1],
                         [0, 0, 0.8870, 0.0089],
                         [0, 0, 0.0089, 0.8870]], dtype=float)

    b_matrix = np.array([[1, 0],
                         [0, 0],
                         [1, 0],
                         [0, 1]], dtype=float)

    (t_jumps, k_matrices) = find_safe_controller(a_matrix, b_matrix, init_r, usafe_r)
    print(t_jumps, k_matrices)

    # a_matrix_ext, b_matrix_ext = extend_a_b(a_matrix, b_matrix)
    #
    # Q1 = np.array([[1, 0, 0, 0],
    #                [0, 1, 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]], dtype=float)
    #
    # u_dim = len(b_matrix[0])
    # R1 = 0.01 * np.eye(u_dim)
    # k_matrix1 = get_input(a_matrix, b_matrix, Q1, R1)
    #
    # a_bk_matrix1 = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix1)
    # pv_object = run_hylaa(settings, init_r, [a_bk_matrix1], [int(10 / step_size)], usafe_r)
    # lce_object = pv_object.compute_longest_ce(lpi_required=False, control_synth=True)
    #
    # t_jumps = lce_object.switching_times
    # k_matrices = [k_matrix1]
    #
    # Q2 = 0.1*np.array([[0.58, 0, 0, 0],
    #                [0, 1/0.59, 0, 0],
    #                [0, 0, 0.68, 0],
    #                [0, 0, 0, 1/9.08]], dtype=float)
    #
    # R2 = 0.01 * np.eye(u_dim)
    # k_matrix2 = get_input(a_matrix, b_matrix, Q2, R2)
    # k_matrices.append(k_matrix2)
    #
    # Q3 = np.array([[1, 0, 0, 0],
    #                [0, 1, 0, 0],
    #                [0, 0, 1, 0],
    #                [0, 0, 0, 1]], dtype=float)
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
