
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
max_time = 1.5


def define_ha(settings, args):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    usafe_r = args[0]
    x_ref = args[1]
    step_inputs = args[2]
    k_matrix = args[3]
    a_matrices = args[4]
    b_matrices = args[5]
    dim = len(b_matrices[0])

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "t"]

    locations = []
    n_locations = len(step_inputs)
    for idx in range(n_locations):
        loc_name = 'loc' + str(idx)
        loc = ha.new_mode(loc_name)
        a_matrix = a_matrices[idx]
        b_matrix = b_matrices[idx]
        b_k_matrix = np.matmul(b_matrix, k_matrix)
        loc.a_matrix = a_matrix + b_k_matrix
        c_vector = -np.matmul(b_k_matrix, x_ref[idx])
        c_vector = c_vector + np.matmul(b_matrix, step_inputs[idx])
        c_vector[dim-1] = step_size
        # print(c_vector)
        loc.c_vector = c_vector
        loc.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 1.0], step_size*idx))  # t <= 0.1
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans.condition_list.append(LinearConstraint([-0.0, -0.0, 0.0, 0.0, -1.0], -step_size*idx))  # t >= 0.1

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], -3.5))
        # usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0], -3.5))
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

    a_matrix = np.array(
        [[1, 0, 0.1, 0, 0], [0, 1, 0, 0.1, 0], [0, 0, 0.8870, 0.0089, 0], [0, 0, 0.0089, 0.8870, 0], [0, 0, 0, 0, 1]])
    b_matrix = np.array([[0, 0], [0, 0], [1, 0], [0, 1], [0, 0]], dtype=float)

    dim = len(b_matrix)
    l_idx = 0
    k_matrix = []
    controller_f_name = '/home/manishg/Research/realsyn/controller_outputs/' + str(4) + '_controller.txt'
    controller_f = open(controller_f_name, 'r')
    lines = controller_f.readlines()
    k_tokens = ast.literal_eval(lines[l_idx])
    l_idx += 1
    for idx in range(len(k_tokens)):
        k_token = k_tokens[idx]
        k_token.append(0.0)
        k_matrix.append(k_token)
    k_matrix = np.array(k_matrix, dtype=float)
    print(k_matrix)
    print(k_matrix.shape)

    n_iterations = int(lines[l_idx])
    l_idx += 1
    for c_idx in range(n_iterations):
        ref_state = ast.literal_eval(lines[l_idx])
        l_idx += 1
        ref_state.append(0.0)
        ref_state = np.array(ref_state)
        n_inputs = int(lines[l_idx])
        l_idx += 1
        n_locations = n_inputs
        a_matrices = []
        b_matrices = []
        c_vectors = []
        max_steps = []
        step_inputs = []
        for idx in range(n_locations):
            step_input = ast.literal_eval(lines[l_idx])
            step_inputs.append(step_input)
            l_idx += 1
            a_matrices.append(a_matrix)
            b_matrices.append(b_matrix)
            c_vector = np.matmul(b_matrix, step_input)
            c_vector[dim - 1] = step_size
            # print(c_vector)
            c_vectors.append(c_vector)
            max_steps.append(1)
        init_rect = ast.literal_eval(lines[l_idx])
        last_pair = (0.0, 0.0)
        init_rect.append(last_pair)
        print(init_rect)
        l_idx += 1
        ref_simulation = np.array(compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True))
        print(ref_simulation)
        # sim_t = np.array(ref_simulation).T
        # plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
        # plt.show()

        init_r = HyperRectangle(init_rect)

        pv_object = run_hylaa(settings, init_r, None, ref_simulation, step_inputs, k_matrix, a_matrices, b_matrices)
