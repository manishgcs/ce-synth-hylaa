
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
dim = 9
max_time = 1.0


def define_ha(settings, args):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    usafe_r = args[0]
    if len(args) > 2:
        x_ref = args[1]
        step_inputs = args[2]

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "t"]

    a_matrix = np.array([[1, 0.1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0.1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0.1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0.1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0],
                         [0.1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0.1, -0.1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0.1, -0.1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0.1, -0.1],
                         [0, 0, 0, 0]], dtype=float)

    k_matrix = np.array([[-7.62931447, -8.9594415, -1.18402956, -1.25308175, -0.69403546, -0.73487489, -0.32086377, -0.33985575, 0],
                         [-6.44528491, -7.70635974, 6.93527901, 8.2245666, -1.50489333, -1.5929375, -0.69403546, -0.73487489, 0],
                         [-5.75124945, -6.97148485, 6.12442114, 7.36650399, 6.93527901, 8.2245666, -1.18402956, -1.25308175, 0],
                         [-5.43038568, -6.6316291, 5.75124945, 6.97148485, 6.44528491, 7.70635974, 7.62931447, 8.9594415, 0]], dtype=float)

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
        loc.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], step_size*idx))  # t <= 0.1
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans.condition_list.append(LinearConstraint([-0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], -step_size*idx))  # t >= 0.1

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(locations[0], error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1))
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
    rv.append((ha.modes['loc0'], init_r))

    return rv


def define_settings():
    'get the hylaa settings object'
    plot_settings = PlotSettings()
    plot_settings.plot_mode = PlotSettings.PLOT_IMAGE
    plot_settings.xdim = 0
    plot_settings.ydim = 1

    s = HylaaSettings(step=0.1, max_time=max_time, disc_dyn=True, plot_settings=plot_settings)
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
    step_inputs = np.array([[-10.0, 10.0, -10.0, -10.0], [-3.36209170649336, 10.0, 6.4604639493427065, 10.0],
                            [10.0, -10.0, 10.0, -1.6771902821856701], [10.0, -1.9811365210998977, 10.0, 10.0],
                            [2.0988711295558087, 2.0315381630524, 10.0, 10.0], [10.0, -10.0, -10.0, -1.8623457684716231],
                            [10.0, 10.0, -10.0, -10.0], [10.0, -3.1596986511242804, -10.0, -10.0],
                            [-10.0, 0.6627009949648011, 10.0, 10.0], [-10.0, 10.0, 10.0, 10.0]], dtype=float)

    a_matrix = np.array([[1, 0.1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0.1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0],
                         [0.1, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0.1, -0.1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0.1, -0.1, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0.1, -0.1],
                         [0, 0, 0, 0]], dtype=float)

    n_locations = len(step_inputs)
    for idx in range(n_locations):
        a_matrices.append(a_matrix)
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[dim-1] = step_size
        # print(c_vector)
        c_vectors.append(c_vector)
        max_steps.append(1)

    ref_state = np.array([0.0, 20.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    ref_simulation = np.array(compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True))
    print(ref_simulation)
    sim_t = np.array(ref_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    plt.show()

    init_r = HyperRectangle([(-0.10606703436571914, 0.10606703436571914),
                             (19.89393296563428, 20.10606703436572), (0.8939329656342808, 1.106067034365719),
                             (-0.10606703436571914, 0.10606703436571914), (0.8939329656342808, 1.106067034365719),
                             (-0.10606703436571914, 0.10606703436571914), (0.8939329656342808, 1.106067034365719),
                             (-0.10606703436571914, 0.10606703436571914), (0, 0)])

    pv_object = run_hylaa(settings, init_r, None, ref_simulation, step_inputs)
