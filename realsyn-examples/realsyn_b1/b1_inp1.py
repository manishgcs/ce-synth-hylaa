
import numpy as np
from hylaa.hybrid_automaton import LinearHybridAutomaton, HyperRectangle, LinearConstraint
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaSettings, HylaaEngine
from hylaa.containers import PlotSettings
from hylaa.pv_container import PVObject
from hylaa.timerutil import Timers
from hylaa.simutil import compute_simulation
import matplotlib.pyplot as plt
import ast

step_size = 0.1
dim = 3
max_time = 1.0


def define_ha(settings, args):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    usafe_r = args[0]
    if len(args) > 2:
        x_ref = args[1]
        step_inputs = args[2]

    ha = LinearHybridAutomaton()
    ha.variables = ["x", "y", "t"]

    a_matrix = np.array([[0, 2, 0], [1, 0, 0], [0, 0, 1]])
    b_matrix = np.array([[1], [1], [0]], dtype=float)
    # k_matrix = np.array([[-1.5359, -1.9282, 0]], dtype=float)
    k_matrix = np.array([[-1.44160279,  1.0782186, 0]], dtype=float)
    # x_ref = np.array([[0.31646814, 0.31646814, 0.1], [-0.11100583, -0.11100583,  0.2], [0.29369705, 0.29369705, 0.3],
    # [-1.42447362, -1.42447362,  0.4], [-0.8048898, -0.8048898,  0.5], [-1.78614487, -1.78614487,  0.6],
    # [-0.73403495, -0.73403495,  0.7], [0.38997233, 0.38997233, 0.8], [2.38997233, 2.38997233, 0.9],
    # [3.4160241, 3.4160241, 1.]])

    locations = []
    n_locations = len(step_inputs)
    for idx in range(n_locations):
        loc_name = 'loc' + str(idx)
        loc = ha.new_mode(loc_name)
        b_k_matrix = np.matmul(b_matrix, k_matrix)
        loc.a_matrix = a_matrix + b_k_matrix
        c_vector = -np.matmul(b_k_matrix, x_ref[idx])
        c_vector = c_vector + np.matmul(b_matrix, step_inputs[idx])
        c_vector[dim-1] = step_size
        # print(c_vector)
        loc.c_vector = c_vector
        # loc.c_vector = np.array([step_inputs[idx], step_inputs[idx], 1], dtype=float)
        loc.inv_list.append(LinearConstraint([0.0, 0.0, 1.0], step_size*idx))  # t <= 0.1
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans.condition_list.append(LinearConstraint([-0.0, -0.0, -1.0], -step_size*idx))  # t >= 0.1

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        # usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0], -10))
        # usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0, 0.0], -10))
        usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0, 0.0], -2))
    else:
        usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])

        for constraint in usafe_star.constraint_list:
            usafe_set_constraint_list.append(constraint)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], error)
        for constraint in usafe_set_constraint_list:
            trans.condition_list.append(constraint)

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

    s = HylaaSettings(step=step_size, max_time=max_time, disc_dyn=True, plot_settings=plot_settings)
    s.stop_when_error_reachable = False
    
    return s


def run_hylaa(settings, init_r, *args):

    'Runs hylaa with the given settings, returning the HylaaResult object.'
    assert len(args) > 0

    ha, usafe_set_constraint_list = define_ha(settings, args)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)


if __name__ == '__main__':
    settings = define_settings()

    a_matrices = []
    c_vectors = []
    max_steps = []
    a_matrix = np.array([[0, 2, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    b_matrix = np.array([[1], [1], [0]], dtype=float)

    step_inputs = [[-0.6835318568657612], [-0.4274739688077575], [0.4047028719575525], [-1.7181706660550653],
                   [0.6195838154872904], [-0.981255069072019], [1.0521099187388827], [1.1240072822724865],
                   [2.0], [1.0260517738498387]]

    n_locations = len(step_inputs)
    for idx in range(n_locations):
        a_matrices.append(a_matrix)
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[2] = step_size
        # print(c_vector)
        c_vectors.append(c_vector)
        max_steps.append(1)

    ref_state = np.array([1, 1, 0])
    ref_simulation = np.array(compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True))
    print(ref_simulation)
    sim_t = np.array(ref_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    plt.show()

    init_r = HyperRectangle([(0.5757318625371235, 1.4242681374628765), (0.5757318625371235, 1.4242681374628765), (0.0, 0.0)])

    usafe_r = HyperRectangle([(4, 8), (4, 8), (0, 10)])

    pv_object = run_hylaa(settings, init_r, None, ref_simulation, step_inputs)

