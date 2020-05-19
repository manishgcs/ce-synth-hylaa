'''
Damped oscillator model, with dynamics:

x' == -0.1 * x + y
y' == -x - 0.1 * y
'''

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
    ha.variables = ["x", "y"]

    loc1 = ha.new_mode('loc1')

    loc1.a_matrix = np.array([[-0.1, 1], [-1, -0.1]])
    # loc1.a_matrix = np.array([[0, 1], [-1, 0]])
    loc1.c_vector = np.array([2, 0], dtype=float)
    # loc1.set_dynamics(a_matrix, c_vector)
    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(loc1, error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0], 2))
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0], 2))
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0], -1))
        usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0], 6))
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

    s = HylaaSettings(step=0.2, max_time=20.0, disc_dyn=False, plot_settings=plot_settings)
    s.stop_when_error_reachable = False
    
    return s


def run_hylaa(settings, init_r, usafe_r):

    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, usafe_r)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


def compute_simulation_mt(milp_ce, z3_ce):
    a_matrix = np.array([[-0.1, 1], [-1, -0.1]])
    c_vector = np.array([0, 0], dtype=float)
    milp_ce_simulation = compute_simulation(milp_ce, a_matrix, c_vector, 0.2, 100)

    z3_ce_simulation = compute_simulation(z3_ce, a_matrix, c_vector, 0.2, 100)

    with open("simulation", 'w') as f:
        f.write('milp_simulation = [')
        for point in milp_ce_simulation:
            f.write('{},{};\n'.format(str(point[0]), str(point[1])))
        f.write(']')
        f.write('\n**************************************\n')
        f.write('z3_simulation = [')
        for point in z3_ce_simulation:
            f.write('{},{};\n'.format(point[0], str(point[1])))
        f.write(']')


if __name__ == '__main__':
    settings = define_settings()
    init_r = HyperRectangle([(-6, -5), (0, 1)])

    # usafe_r = HyperRectangle([(-1.5, 1.5), (4, 6)])  # Small
    usafe_r = HyperRectangle([(-2, 2), (1, 5)])   # milp
    # usafe_r = HyperRectangle([(-3, 2), (1, 5)])  # Medium
    # usafe_r = HyperRectangle([(-4, 2), (-1, 6)])  # Large

    pv_object = run_hylaa(settings, init_r, usafe_r)

    # pv_object.compute_longest_ce()
    # depth_direction = np.identity(len(init_r.dims))
    # pv_object.compute_deepest_ce(depth_direction[1])
    robust_pt = pv_object.compute_robust_ce()

    # pv_object.compute_counter_examples_using_z3(4)
    # pv_object.compute_z3_counterexample()
    # pv_object.compute_milp_counterexample_py('Oscillator', "11111001111111111111111110")
    # z3_ce = np.array([-5.1324054061, 0.8009245168])
    # milp_ce = np.array([-5.28179, 0.76464])
    # compute_simulation_mt(milp_ce, z3_ce)

    # simulations = []
    # for ce in z3_counter_examples:
    # simulation = compute_simulation(z3_counter_example, a_matrix, c_vector, 0.2, 20/0.2)
    # x, y = np.array(simulation).T
    # plt.plot(x, y, 'r*')
    # verts = usafe_star.verts()
    # print(verts)
    # x, y = np.array(verts).T
    # plt.plot(x, y, 'r-', robust_pt[0], robust_pt[1], 'r.')
    # plt.show()
    Timers.print_stats()
