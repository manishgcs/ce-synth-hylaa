
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
dim = 13
max_time = 1.0


def define_ha(settings, args):
    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    usafe_r = args[0]
    if len(args) > 2:
        x_ref = args[1]
        step_inputs = args[2]

    ha = LinearHybridAutomaton()
    ha.variables = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "t"]

    a_matrix = np.array([[1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0.8870, 0.0089, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0.0089, 0.8870, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0.1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.8870, 0.0089, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.0089, 0.8870, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8870, 0.0089, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0089, 0.8870, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0]], dtype=float)

    k_matrix = np.array(
        [[-6.06817591e-01, 1.34582417e-03, -6.21076818e-01, -6.93694811e-03, -3.87285782e-16, 3.94330035e-16, -6.83624636e-17, 6.75996315e-17, 1.35987476e-16, -2.75721049e-18, 1.76212816e-17, 8.73603657e-18, 0],
        [1.34582417e-03, -6.06817591e-01, -6.93694811e-03, -6.21076818e-01, -1.46903336e-16, 1.32989521e-16, -2.31736641e-17, -1.71600848e-17, -1.08498995e-16, 2.14839550e-17, -1.41359722e-17, -4.30368678e-18, 0],
        [-6.98203533e-16, -1.48858830e-16, -9.98219318e-17, -2.33646234e-17, -6.06817591e-01, 1.34582417e-03, -6.21076818e-01, -6.93694811e-03, -2.04810607e-15, -1.68505655e-16, -3.90716348e-16, -1.37933658e-17, 0],
        [5.70185336e-16, -6.46528547e-16, 8.51805715e-17, -9.47441985e-17, 1.34582417e-03, -6.06817591e-01, -6.93694811e-03, -6.21076818e-01, -1.16635931e-16, 3.57991501e-16, -1.78718942e-17, 7.64404029e-17, 0],
        [6.48295567e-17, -5.92654958e-17, 1.03831377e-17, -9.10874950e-18, -1.96276622e-15, 7.73637815e-17, -3.82298152e-16, -7.46216907e-19, -6.06817591e-01, 1.34582417e-03, -6.21076818e-01, -6.93694811e-03, 0],
        [2.17336814e-16, -1.95215818e-16, 3.06415662e-17, -2.58513121e-17, -2.34811479e-16, -4.85104150e-16, -1.81496543e-17, -7.75337395e-18, 1.34582417e-03, -6.06817591e-01, -6.93694811e-03, -6.21076818e-01, 0]],
        dtype=float)

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
        loc.inv_list.append(LinearConstraint([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], step_size*idx))  # t <= 0.1
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans.condition_list.append(LinearConstraint([-0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], -step_size*idx))  # t >= 0.1

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(locations[0], error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1))
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
    step_inputs = np.array([[-5.0, 5.0, -5.0, -0.13077844393183274, 5.0, 5.0],
                   [5.0, 5.0, -3.5883416133978283, 5.0, -5.0, 5.0], [-4.809987544741765, 5.0, 5.0, 5.0, 0.1620858597640832, 5.0],
                   [5.0, 5.0, 5.0, 5.0, 5.0, 5.0], [-2.7836139525625687, 5.0, 5.0, 5.0, -5.0, 4.071582652549173],
                   [5.0, -5.0, -5.0, 5.0, -0.6335312617843153, -4.810524802755337], [5.0, 1.4606184068318073, -2.152778547792834, -4.087936570624512, -1.8627102438238172, 2.474137973091641],
                   [5.0, -5.0, 5.0, -5.0, -5.0, -5.0], [5.0, -5.0, 5.0, -5.0, -5.0, -4.912434473005285],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=float)

    a_matrix = np.array([[1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0.8870, 0.0089, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0.0089, 0.8870, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0.1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.8870, 0.0089, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0.0089, 0.8870, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.8870, 0.0089, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0089, 0.8870, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=float)

    b_matrix = np.array([[0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [1, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 1],
                         [0, 0, 0, 0, 0, 0]], dtype=float)

    n_locations = len(step_inputs)
    for idx in range(n_locations):
        a_matrices.append(a_matrix)
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[dim-1] = step_size
        # print(c_vector)
        c_vectors.append(c_vector)
        max_steps.append(1)

    ref_state = np.array([-4.0, -4.0, 0.0, 0.0, -2.0, -4.0, 0.0, 0.0, 2.0, -4.0, 0.0, 0.0, 0.0])
    ref_simulation = np.array(compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True))
    print(ref_simulation)
    sim_t = np.array(ref_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    plt.show()

    init_r = HyperRectangle([(-4.1270171184434625, -3.872982881556537), (-4.1270171184434625, -3.872982881556537),
                             (-0.12701711844346295, 0.12701711844346295), (-0.12701711844346295, 0.12701711844346295),
                             (-2.127017118443463, -1.872982881556537), (-4.1270171184434625, -3.872982881556537),
                             (-0.12701711844346295, 0.12701711844346295), (-0.12701711844346295, 0.12701711844346295),
                             (1.872982881556537, 2.127017118443463), (-4.1270171184434625, -3.872982881556537),
                             (-0.12701711844346295, 0.12701711844346295), (-0.12701711844346295, 0.12701711844346295),
                             (0, 0)])

    pv_object = run_hylaa(settings, init_r, None, ref_simulation, step_inputs)
