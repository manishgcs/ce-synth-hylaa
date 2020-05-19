
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
dim = 5
max_time = 1.0


def define_ha(settings, args):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    usafe_r = args[0]
    if len(args) > 2:
        x_ref = args[1]
        step_inputs = args[2]

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "t"]

    a_matrix = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    b_matrix = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1], [0, 0, 0, 0]], dtype=float)

    k_matrix = np.array([[-0.95124922, 0, 0, 0, 0],
                         [0, -0.95124922, 0, 0, 0],
                         [0, 0, -0.95124922, 0, 0],
                         [0, 0, 0, -0.95124922, 0]], dtype=float)
    print(k_matrix.shape)

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
        loc.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 1.0], step_size*idx))  # t <= 0.1
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans.condition_list.append(LinearConstraint([-0.0, -0.0, 0.0, 0.0, -1.0], -step_size*idx))  # t >= 0.1

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_r is None:
        # usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, 0.0, 0.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0, 0.0, 0.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0, 0.0, 0.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, 1.0, 0.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, -1.0, 0.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, 0.0, 1.0, 0.0], -2))
        # usafe_set_constraint_list.append(LinearConstraint([0.0, 0.0, 0.0, -1.0, 0.0], -2))
        usafe_set_constraint_list.append(LinearConstraint([1.0, 0.0, -1.0, 0.0, 0.0], 0.5))
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 1.0, 0.0, 0.0], 0.5))
        usafe_set_constraint_list.append(LinearConstraint([0.0, 1.0, 0.0, -1.0, 0.0], 0.5))
        usafe_set_constraint_list.append(LinearConstraint([0.0, -1.0, 0.0, 1.0, 0.0], 0.5))
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
    step_inputs = [[-6.121190010192605, 10.0, 10.0, 10.0],
                   [0.2776591143039765, 10.0, -4.156469104111374, 7.893784909297777],
                   [0.26412301583120223, 10.0, -0.26412301583120223, 10.0],
                   [0.25124681272085664, 10.0, 9.509754852720027, 10.0],
                   [0.23899833455911693, 10.0, -10.0, 10.0],
                   [0.22734697926494332, 9.112778768204867, -0.22734697926494332, 10.0],
                   [10.0, 10.0, -10.0, 10.0], [10.0, -4.672947266926665, -10.0, 10.0],
                   [1.2295368799380306, 9.192447372296275, -10.0, 8.801064222151364],
                   [10.0, 10.0, -1.2295368799380313, -3.0625702578746608]]

    a_matrix = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]])
    b_matrix = np.array([[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0.1, 0], [0, 0, 0, 0.1], [0, 0, 0, 0]], dtype=float)

    n_locations = len(step_inputs)
    for idx in range(n_locations):
        a_matrices.append(a_matrix)
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[dim-1] = step_size
        # print(c_vector)
        c_vectors.append(c_vector)
        max_steps.append(1)

    ref_state = np.array([-2.0, -4.0, 2.0, -4.0, 0.0])
    ref_simulation = np.array(compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True))
    # print(ref_simulation)
    sim_t = np.array(ref_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    plt.show()

    init_r = HyperRectangle([(-2.299368469619189, -1.7006315303808108), (-4.299368469619189, -3.700631530380811),
                             (1.7006315303808108, 2.299368469619189), (-4.299368469619189, -3.700631530380811), (0, 0)])

    pv_object = run_hylaa(settings, init_r, None, ref_simulation, step_inputs)

