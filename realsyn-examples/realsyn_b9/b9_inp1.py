
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
dim = 17
max_time = 1.0


def define_ha(settings, args):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    usafe_r = args[0]
    if len(args) > 2:
        x_ref = args[1]
        step_inputs = args[2]

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "t"]

    a_matrix = np.array([[1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                         [0.1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0.1, -0.1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0.1, -0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0.1, -0.1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0.1, -0.1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0.1, -0.1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0.1, -0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.1, -0.1],
                         [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)

    k_matrix = np.array(
        [[-7.39852746, -8.71218232, -1.43354481, -1.52015133, -0.98386583, -1.04458066,
  -0.67892718, -0.721701,   -0.46540552, -0.49529003, -0.30957388, -0.32977631,
  -0.18945168, -0.20196845, -0.08999092, -0.09598254, 0],
 [-5.96498265, -7.19203099,  6.41466163,  7.66760165, -2.11247199, -2.24185234,
  -1.44927135, -1.53987069, -0.98850106, -1.05147731, -0.65485719, -0.69725848,
  -0.3995648,  -0.42575885, -0.18945168, -0.20196845, 0],
 [-4.98111682, -6.14745032,  5.28605547,  6.47032998,  5.94925611,  7.17231163,
  -2.42204586, -2.57162864, -1.63872302, -1.74183914, -1.07849198, -1.14745985,
  -0.65485719, -0.69725848, -0.30957388, -0.32977631, 0],
 [-4.30218964, -5.42574932,  4.5157113,   5.65216029,  4.97648159,  6.14055368,
   5.75980443,  6.97034317, -2.51203678, -2.66761118, -1.63872302, -1.74183914,
  -0.98850106, -1.05147731, -0.46540552, -0.49529003, 0],
 [-3.83678412, -4.93045929,  3.99261576,  5.09597301,  4.32625963,  5.45019184,
   4.88649067,  6.04457114,  5.75980443,  6.97034317, -2.42204586, -2.57162864,
  -1.44927135, -1.53987069, -0.67892718, -0.721701, 0],
 [-3.52721025, -4.60068298,  3.64733245,  4.72849084,  3.90262484,  4.99999047,
   4.32625963,  5.45019184,  4.97648159,  6.14055368,  5.94925611,  7.17231163,
  -2.11247199, -2.24185234, -0.98386583, -1.04458066, 0],
 [-3.33775857, -4.39871453,  3.43721933,  4.50470044,  3.64733245,  4.72849084,
  3.99261576,  5.09597301,  4.5157113,  5.65216029,  5.28605547,  6.47032998,
   6.41466163,  7.66760165, -1.43354481, -1.52015133, 0],
 [-3.24776765, -4.30273199,  3.33775857,  4.39871453,  3.52721025,  4.60068298,
   3.83678412,  4.93045929,  4.30218964,  5.42574932,  4.98111682,  6.14745032,
   5.96498265,  7.19203099,  7.39852746,  8.71218232, 0]], dtype=float)

    locations = []
    n_locations = len(step_inputs)-4
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
        loc.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], step_size*idx))  # t <= 0.1
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans.condition_list.append(LinearConstraint([-0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], -step_size*idx))  # t >= 0.1

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(locations[0], error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1))
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
    step_inputs = np.array([[-10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, -1.5287986568113925],
                   [-10.0, -8.543561347687453, 10.0, -10.0, -6.228855613257126, -10.0, -10.0, -10.0], [10.0, 10.0, -10.0, -10.0, -10.0, -0.3565219422739816, -10.0, -2.6354125011883527],
                   [10.0, -10.0, -10.0, 0.9981401622992315, 3.9181533287162775, -2.638035556726259, -10.0, -10.0], [10.0, 8.543561347687453, -10.0, 10.0, -10.0, -10.0, -6.668827084457414, 10.0],
                   [10.0, -2.3281235836443894, -0.3367130394054635, -2.56998394227622, 10.0, -10.0, 10.0, 10.0], [10.0, -10.0, 10.0, 10.0, -10.0, 10.0, 10.0, -10.0],
                   [-9.006005624417782, 2.4001880499227877, -0.442854710641922, -10.0, 10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 3.0536401600793175, 10.0],
                   [-10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]], dtype=float)

    a_matrix = np.array([[1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[0,   0,   0,   0,   0,   0,   0,   0],
        [0.1, 0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0.1, -0.1, 0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0.1, -0.1, 0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,  0.1, -0.1, 0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0.1, -0.1, 0,   0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0.1, -0.1, 0,   0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,  0.1, -0.1, 0],
        [0,   0,   0,   0,   0,   0,   0,   0],
        [0,   0,   0,   0,   0,   0, 0.1, -0.1],
        [0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)

    n_locations = len(step_inputs)
    for idx in range(n_locations):
        a_matrices.append(a_matrix)
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[dim-1] = step_size
        # print(c_vector)
        c_vectors.append(c_vector)
        max_steps.append(1)

    ref_state = np.array([0.0, 20.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    ref_simulation = np.array(compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True))
    print(ref_simulation)
    sim_t = np.array(ref_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    plt.show()

    init_r = HyperRectangle([(-0.075, 0.075), (19.925, 20.075), (0.925, 1.075), (-0.075, 0.075), (0.925, 1.075),
                             (-0.075, 0.075), (0.925, 1.075), (-0.075, 0.075), (0.925, 1.075), (-0.075, 0.075),
                             (0.925, 1.075), (-0.075, 0.075), (0.925, 1.075), (-0.075, 0.075), (0.925, 1.075),
                             (-0.075, 0.075), (0, 0)])

    pv_object = run_hylaa(settings, init_r, None, ref_simulation, step_inputs)
