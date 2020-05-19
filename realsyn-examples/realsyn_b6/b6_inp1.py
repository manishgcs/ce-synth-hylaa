
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

    a_matrix = np.array([[1, 0, 0.1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0.8870, 0.0089, 0, 0, 0, 0, 0],
                         [0, 0, 0.0089, 0.8870, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0.1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0.8870, 0.0089, 0],
                         [0, 0, 0, 0, 0, 0, 0.0089, 0.8870, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 0, 0],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1],
                         [0, 0, 0, 0]], dtype=float)

    k_matrix = np.array([[-6.06817591e-01, 1.34582417e-03, -6.21076818e-01, -6.93694811e-03, -9.67329920e-16, -7.52982545e-17, -1.54021839e-16, -7.58172674e-18, 0],
                        [1.34582417e-03, -6.06817591e-01, -6.93694811e-03, -6.21076818e-01, -9.56139051e-17, -6.49820163e-16, -2.11619731e-17, -9.20008924e-17, 0],
                        [-6.44795721e-16, 1.07029473e-17, -1.21887356e-16, -1.08352047e-17, -6.06817591e-01, 1.34582417e-03, -6.21076818e-01, -6.93694811e-03, 0],
                        [4.96602732e-17, -5.76168579e-16, 5.21904291e-18, -8.45167971e-17, 1.34582417e-03, -6.06817591e-01, -6.93694811e-03, -6.21076818e-01, 0]], dtype=float)

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
    step_inputs = np.array([[4.980899118147371, -0.41299540713592886, 5.0, 4.670680931365026],
                   [5.0, 5.0, 3.2846654241217768, 5.0], [5.0, -0.7380710757211743, 5.0, 5.0],
                   [-0.3227161432083178, 5.0, -5.0, 5.0], [-5.0, 5.0, -5.0, 5.0],
                   [-4.978773616725495, 5.0, -5.0, -5.0], [-5.0, 5.0, -5.0, -5.0],
                   [-5.0, -5.0, -4.9459409617826, -1.9006819084484023], [-5.0, -4.092170848190299, -5.0, 5.0],
                   [0.0, 0.0, 0.0, 0.0]], dtype=float)

    a_matrix = np.array([[1, 0, 0.1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0.8870, 0.0089, 0, 0, 0, 0, 0],
                         [0, 0, 0.0089, 0.8870, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0.1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0.8870, 0.0089, 0],
                         [0, 0, 0, 0, 0, 0, 0.0089, 0.8870, 0],
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

    n_locations = len(step_inputs)
    for idx in range(n_locations):
        a_matrices.append(a_matrix)
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[dim-1] = step_size
        # print(c_vector)
        c_vectors.append(c_vector)
        max_steps.append(1)

    ref_state = np.array([-2.0, -4.0, 0.0, 0.0, 2.0, -4.0, 0.0, 0.0, 0.0])
    ref_simulation = np.array(compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True))
    print(ref_simulation)
    sim_t = np.array(ref_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    plt.show()

    init_r = HyperRectangle([(-2.106067034365719, -1.893932965634281), (-4.1060670343657195, -3.893932965634281),
                             (-0.10606703436571914, 0.10606703436571914), (-0.10606703436571914, 0.10606703436571914),
                             (1.893932965634281, 2.106067034365719), (-4.1060670343657195, -3.893932965634281),
                             (-0.10606703436571914, 0.10606703436571914), (-0.10606703436571914, 0.10606703436571914),
                             (0, 0)])

    pv_object = run_hylaa(settings, init_r, None, ref_simulation, step_inputs)
