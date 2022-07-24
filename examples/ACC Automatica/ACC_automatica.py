'''
Adaptive cruise control, with dynamics from paper by Tiwari
'''

import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    # k = -0.0025  # Unsafe
    # k = -1.5  # Safe
    ha = LinearHybridAutomaton()
    ha.variables = ["s", "v", "vf", "a", "t"]

    loc1 = ha.new_mode('loc1')

    # loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -4, 3, -3, 0], [0, 0, 0, 0, 0]])
    loc1.a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -4, 3, -1, 0], [0, 0, 0, 0, 0]])
    loc1.c_vector = np.array([0, 0, 0, -10, 1], dtype=float)

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([0, 1, 0, 0, 0], 15))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for constraint in usafe_set_constraint_list:
        trans.condition_list.append(constraint)

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

    s = HylaaSettings(step=1, max_time=3.0, plot_settings=plot_settings, disc_dyn=False)
    s.stop_when_error_reachable = False

    return s


def run_hylaa(settings, init_r, usafe_r):
    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


def compute_simulation_mt(longest_ce, deepest_ce=None):
    a_matrix = np.array([[0, -1, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [1, -4, 3, -1.2, 0], [0, 0, 0, 0, 0]])
    c_vector = np.array([0, 0, 0, -10, 1], dtype=float)
    longest_ce_simulation = compute_simulation(longest_ce, a_matrix, c_vector, 0.1, 200)

    deepest_ce_simulation = compute_simulation(deepest_ce, a_matrix, c_vector, 0.1, 200)

    with open("simulation", 'w') as f:
        f.write(' Unsafe longest_simulation = [')
        t = 0.0
        for point in longest_ce_simulation:
            f.write('{},{};\n'.format(str(point[4]), str(point[1])))
            t = t + 0.1
        f.write('];')
        f.write('\n**************************************\n')
        f.write('deepest_simulation = [')
        t = 0.0
        for point in deepest_ce_simulation:
            f.write('{},{};\n'.format(point[4], str(point[1])))
            t = t + 0.1
        f.write('];')


if __name__ == '__main__':
    settings = define_settings()
    # init_r = HyperRectangle([(-9, -2), (18, 19), (20, 20), (-1, 1), (0, 0)])
    init_r = HyperRectangle([(2, 5), (18, 22), (20, 20), (-1, 1), (0, 0)])

    pv_object = run_hylaa(settings, init_r, None)
    Timers.print_stats()