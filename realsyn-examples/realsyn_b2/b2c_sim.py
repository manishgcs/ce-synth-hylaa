
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

step_size = 0.05


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "t"]

    loc1 = ha.new_mode('loc1')

    a_matrix = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]], dtype=float)

    # Q = 5 * np.eye(2)
    Q1 = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)

    u_dim = len(b_matrix[0])

    R = 0.1*np.eye(u_dim)
    (X1, L, G) = care(a_matrix, b_matrix, Q1, R)
    G = np.array(G)
    k_matrix1 = []
    for idx in range(len(G)):
        k_token = G[idx].tolist()
        k_token.append(0.0)
        k_matrix1.append(k_token)
    k_matrix1 = np.array(k_matrix1, dtype=float)

    loc2 = ha.new_mode('loc2')
    Q2 = np.array([[100, 0, 0, 0],
                   [0, 140, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 140]], dtype=float)

    (X2, L, G) = care(a_matrix, b_matrix, Q2, R)
    G = np.array(G)
    k_matrix2 = []
    for idx in range(len(G)):
        k_token = G[idx].tolist()
        k_token.append(0.0)
        k_matrix2.append(k_token)
    k_matrix2 = np.array(k_matrix2, dtype=float)
    print(Q2 - Q1)
    print(X2 - X1)
    print(k_matrix2 - k_matrix1)

    a_matrix = np.array([[1, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0],
                         [0, 0, 1, 0, 0],
                         [0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0]], dtype=float)

    b_matrix = np.array([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]], dtype=float)

    a_bk_matrix1 = a_matrix - np.matmul(b_matrix, k_matrix1)
    a_bk_matrix2 = a_matrix - np.matmul(b_matrix, k_matrix2)
    print(a_bk_matrix1)
    print(a_bk_matrix2)

    loc1.a_matrix = a_bk_matrix1
    loc1.c_vector = np.array([0, 0, 0, 0, 1], dtype=float)
    loc1.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 1.0], step_size * 4))  # t <= 0.4

    loc2.a_matrix = a_bk_matrix2
    loc2.c_vector = np.array([0, 0, 0, 0, 1], dtype=float)
    loc2.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, -1.0], -step_size * 4))  # t <= 0.4

    trans1 = ha.new_transition(loc1, loc2)
    trans1.condition_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, -1.0], -step_size * 5))  # t >= 0.4

    error = ha.new_mode('_error')
    error.is_error = True

    trans1_u = ha.new_transition(loc1, error)
    trans2_u = ha.new_transition(loc2, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0, 0, 0], 1.0))
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0, 0, 0], -0.25))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])

        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for constraint in usafe_set_constraint_list:
        trans1_u.condition_list.append(constraint)
        trans2_u.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []
    rv.append((ha.modes['loc1'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    s = HylaaSettings(step=step_size, max_time=10.0, disc_dyn=False, plot_settings=plot_settings)
    s.stop_when_error_reachable = False
    
    return s


def run_hylaa(settings, init_r, usafe_r):

    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-2.5, -2.5), (-4.5, -4.5), (1.5, 1.5), (-4.5, -4.5), (0, 0)])
    # usafe_r = HyperRectangle([(0.1, 0.2), (0.0, 0.1), (0.0, 10.0)])

    pv_object = run_hylaa(settings, init_r, None)
    depth_direction = np.identity(len(init_r.dims))
    for idx in range(len(init_r.dims)-1):
        deepest_ce_1 = pv_object.compute_deepest_ce(depth_direction[idx])
        # print(deepest_ce_1.ce_depth)
        deepest_ce_2 = pv_object.compute_deepest_ce(-depth_direction[idx])
        print("depth difference: {}".format(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth)))
        # print(deepest_ce_2.ce_depth)
    pv_object.compute_longest_ce(lpi_required=False)
    # pv_object.compute_robust_ce()
