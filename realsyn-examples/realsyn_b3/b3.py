
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt


def define_ha(settings, usafe_r):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4"]

    loc1 = ha.new_mode('loc1')

    a_matrix = np.array([[1, 0, 0.1, 0],
             [0, 1, 0, 0.1],
             [0, 0, 0.8870, 0.0089],
             [0, 0, 0.0089, 0.8870]], dtype=float)

    loc1.c_vector = np.array([0, 0, 0, 0], dtype=float)

    b_matrix = np.array([[1, 0],
                        [0, 0],
                        [1, 0],
                        [0, 1]], dtype=float)

    k_matrix = np.array([[24.2999, 2.2873, -19.7882, 0.0153],
                        [0.0771, 46.7949, -0.0618, 4.2253]], dtype=float)

    loc1.a_matrix = a_matrix - np.matmul(b_matrix, k_matrix)

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

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

    s = HylaaSettings(step=0.02, max_time=10.0, plot_settings=plot_settings)
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
    init_r = HyperRectangle([(0.3, 0.7), (1.3, 1.7), (0, 0), (0, 0)])

    usafe_r = HyperRectangle([(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.5, 1.0)])

    pv_object = run_hylaa(settings, init_r, usafe_r)

