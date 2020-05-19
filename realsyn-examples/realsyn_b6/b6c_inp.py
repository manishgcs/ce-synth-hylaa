
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt

step_size = 0.1


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "t"]

    a_matrix = np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, -1.2, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0.1, -1.2, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, -1.2, -0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0.1, -1.2, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]], dtype=float)

    k_matrix = np.array([[1.0000, -0.0000, 0.9087, 0.0431, -0.0000, -0.0000, -0.0000, -0.0000, 0],
                         [-0.0000, 1.0000, 0.0431, 0.9087, -0.0000, 0.0000, -0.0000, -0.0000, 0],
                         [0.0000, -0.0000, -0.0000, -0.0000, 1.0000, -0.0000, 0.9087, 0.0431, 0],
                         [0.0000, -0.0000, -0.0000, -0.0000, 0.0000, 1.0000, 0.0431, 0.9087, 0]], dtype=float)

    step_inputs = [[4.980899118147371, -0.41299540713592886, 5.0, 4.670680931365026],
                   [5.0, 5.0, 3.2846654241217768, 5.0],
                   [5.0, -0.7380710757211743, 5.0, 5.0],
                   [-0.3227161432083178, 5.0, -5.0, 5.0],
                   [-5.0, 5.0, -5.0, 5.0],
                   [-4.978773616725495, 5.0, -5.0, -5.0],
                   [-5.0, 5.0, -5.0, -5.0],
                   [-5.0, -5.0, -4.9459409617826, -1.9006819084484023],
                   [-5.0, -4.092170848190299, -5.0, 5.0],
                   [0.0, 0.0, 0.0, 0.0]]
    locations = []
    n_locations = len(step_inputs)
    for idx in range(n_locations):
        loc_name = 'loc' + str(idx)
        loc = ha.new_mode(loc_name)
        loc.a_matrix = a_matrix - np.matmul(b_matrix, k_matrix)
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[len(ha.variables) - 1] = 1
        loc.c_vector = c_vector
        # loc.c_vector = np.array([step_inputs[idx], step_inputs[idx], 1], dtype=float)
        loc.inv_list.append(
            LinearConstraint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], step_size * (idx)))  # t <= 0.1
        locations.append(loc)

    for idx in range(n_locations - 1):
        trans = ha.new_transition(locations[idx], locations[idx + 1])
        # trans.condition_list.append(LinearConstraint([-0.0, -0.0, 1.0], step_size * (idx+1)))  # t >= 0.1
        trans.condition_list.append(
            LinearConstraint([-0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], -step_size * (idx)))  # t >= 0.1

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(locations[0], error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0], 2))
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

    s = HylaaSettings(step=0.1, max_time=1.0, disc_dyn=False, plot_settings=plot_settings)
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
    init_r = HyperRectangle([(-2.1, -1.9), (-4.1, -3.9), (0, 0), (0, 0), (1.9, 2.1), (-4.1, -3.9), (0, 0), (0, 0), (0.0, 0.0)])

    usafe_r = HyperRectangle([(-1.0, 1.0), (-1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (-1.0, 1.0), (-1.0, 1.0), (0.0, 0.0), (0.0, 0.0), (0, 1)])

    pv_object = run_hylaa(settings, init_r, usafe_r)

