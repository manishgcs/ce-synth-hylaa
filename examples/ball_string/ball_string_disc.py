'''
The first two modes of Goran's ball-string example. This is a simple model demonstrating guards and transitions.
'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.engine import HylaaSettings
from hylaa.engine import HylaaEngine
from hylaa.plotutil import PlotSettings
from hylaa.star import init_hr_to_star
from hylaa.timerutil import Timers
from hylaa.pv_container import PVObject
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r=None):
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "v"]

    extension = ha.new_mode('extension')
    extension.a_matrix = np.array([[0.9951, 0.0098], [-0.9786, 0.9559]], dtype=float)
    extension.c_vector = np.array([-0.0005, -0.0960], dtype=float)
    extension.inv_list.append(LinearConstraint([1.0, 0.0], 0))  # x <= 0

    freefall = ha.new_mode('freefall')
    freefall.a_matrix = np.array([[1.0, 0.01], [0.0, 1.0]], dtype=float)
    freefall.c_vector = np.array([-0.0005, -0.0981], dtype=float)
    freefall.inv_list.append(LinearConstraint([-1.0, 0.0], 0.0))  # 0 <= x
    freefall.inv_list.append(LinearConstraint([1.0, 0.0], 1.0))  # x <= 1

    trans = ha.new_transition(extension, freefall)
    trans.condition_list.append(LinearConstraint([-0.0, -1.0], -0.0))  # v >= 0

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0], 0.5))
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0], -5.0))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    trans1 = ha.new_transition(extension, error)
    for constraint in usafe_set_constraint_list:
        trans1.condition_list.append(constraint)

    trans2 = ha.new_transition(freefall, error)
    for constraint in usafe_set_constraint_list:
        trans2.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, v]
    rv = [(ha.modes['extension'], init_r)]

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    settings = HylaaSettings(step=0.01, max_time=2.0, disc_dyn=True, plot_settings=plot_settings)
    settings.stop_when_error_reachable = False
    
    return settings


def run_hylaa(settings, init_r, usafe_r):
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)

    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


def compute_simulation_py(longest_ce, deepest_ce, robust_ce):
    step = 0.02
    a_matrix = np.array([[0.0, 1.0], [-100.0, -4.0]], dtype=float)
    c_vector = np.array([0.0, -9.81], dtype=float)
    deepest_simulation = compute_simulation(deepest_ce, a_matrix, c_vector, step, 0.2 / step)
    longest_simulation = compute_simulation(longest_ce, a_matrix, c_vector, step, 0.2 / step)
    robust_simulation = compute_simulation(robust_ce, a_matrix, c_vector, step, 0.2 / step)
    sim_t = np.array(deepest_simulation).T
    deepest_ce_handle = plt.plot(sim_t[0], sim_t[1], 'g', label='Deepest Counterexample=[-0.99, 0.15]', linestyle='--',
                                 linewidth='2')
    sim_t = np.array(longest_simulation).T
    longest_ce_handle = plt.plot(sim_t[0], sim_t[1], 'r', label='Longest Counterexample=[-1.05, -0.15]', linestyle='-.',
                                 linewidth='2')
    sim_t = np.array(robust_simulation).T
    robust_ce_handle = plt.plot(sim_t[0], sim_t[1], 'b', label='Robust Counterexample=[-1.026, 0.0]', linestyle='--',
                                linewidth='2')
    init_state_in_freefall = deepest_simulation[len(deepest_simulation) - 1]
    a_matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    c_vector = np.array([0.0, -9.81], dtype=float)
    deepest_simulation = compute_simulation(init_state_in_freefall, a_matrix, c_vector, step, 0.2 / step)
    sim_t = np.array(deepest_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'g', linestyle='--')

    init_state_in_freefall = longest_simulation[len(longest_simulation) - 1]
    a_matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    c_vector = np.array([0.0, -9.81], dtype=float)
    longest_simulation = compute_simulation(init_state_in_freefall, a_matrix, c_vector, step, 0.2 / step)
    sim_t = np.array(longest_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'r', linestyle='-.')

    init_state_in_freefall = robust_simulation[len(robust_simulation) - 1]
    a_matrix = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=float)
    c_vector = np.array([0.0, -9.81], dtype=float)
    robust_simulation = compute_simulation(init_state_in_freefall, a_matrix, c_vector, step, 0.2 / step)
    sim_t = np.array(robust_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    usafe_vertices = np.array([[-0.5, 5], [0.5, 5], [0.5, 6.2], [-0.5, 6.2], [-0.5, 5]])
    usafe_region_handle = plt.plot(usafe_vertices.T[0], usafe_vertices.T[1],
                                   label='Unsafe Region=([-0.5,0.5],[5.0,6.2])', linestyle='--', linewidth='3',
                                   color='crimson')
    plt.legend(handles=[longest_ce_handle[0], deepest_ce_handle[0], robust_ce_handle[0], usafe_region_handle[0]])
    plt.show()


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-1.05, -0.95), (-0.15, 0.15)])
    # usafe_r = HyperRectangle([(-0.2, 0.2), (5, 6)]) #Small
    # usafe_r = HyperRectangle([(-0.5, 0.5), (5, 6.4)]) #To check

    usafe_r = HyperRectangle([(-0.5, 0.5), (5, 7)])  # Medium
    # usafe_r = HyperRectangle([(-0.5, 0.5), (5, 6.4)])  # Medium (ACC)
    # usafe_r = HyperRectangle([(-0.8, 0.8), (3, 7)])  # Large

    pv_object = run_hylaa(settings, init_r, usafe_r)

    # z3_counter_examples = pv_object.compute_counter_examples_using_z3(2)
    # pv_object.compute_z3_counterexamples()
    # pv_object.compute_milp_counterexamples_py('Ball')

    longest_ce = pv_object.compute_longest_ce()
    # robust_ce = pv_object.compute_robust_ce()
    # depth_direction = np.identity(len(init_r.dims))
    # deepest_ce = pv_object.compute_deepest_ce(depth_direction[1])

    Timers.print_stats()
