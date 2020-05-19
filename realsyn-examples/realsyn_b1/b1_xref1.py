
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
    ha.variables = ["x", "y", "t"]

    step_inputs = [[-0.6835318568657612], [-0.4274739688077575], [0.4047028719575525], [-1.7181706660550653],
                   [0.6195838154872904], [-0.981255069072019], [1.0521099187388827], [1.1240072822724865],
                   [2.0], [1.0260517738498387]]
    a_matrix = np.array([[0, 2, 0], [1, 0, 0], [0, 0, 1]])
    b_matrix = np.array([[1], [1], [0]], dtype=float)

    locations = []
    n_locations = len(step_inputs)
    for idx in range(n_locations):
        loc_name = 'loc' + str(idx)
        loc = ha.new_mode(loc_name)
        loc.a_matrix = a_matrix
        c_vector = np.matmul(b_matrix, step_inputs[idx])
        c_vector[len(ha.variables) - 1] = step_size
        print(c_vector)
        loc.c_vector = c_vector
        # loc.c_vector = np.array([step_inputs[idx], step_inputs[idx], 1], dtype=float)
        loc.inv_list.append(LinearConstraint([0.0, 0.0, 1.0], step_size*idx))
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans.condition_list.append(LinearConstraint([-0.0, -0.0, -1.0], -idx*step_size))

    error = ha.new_mode('_error')
    error.is_error = True

    trans = ha.new_transition(locations[0], error)

    usafe_set_constraint_list = []
    if usafe_r is None:
        usafe_set_constraint_list.append(LinearConstraint([-1.0, 0.0, 0.0], 1))
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

    s = HylaaSettings(step=step_size, max_time=1.0, disc_dyn=True, plot_settings=plot_settings)
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
    # init_r = HyperRectangle([(1.0, 1.0), (1.0, 1.0), (0.0, 0.0)])

    # pv_object = run_hylaa(settings, init_r, None)

    a_matrices = []
    c_vectors = []
    max_steps = []
    step_inputs = [[-0.6835318568657612], [-0.4274739688077575], [0.4047028719575525], [-1.7181706660550653],
                   [0.6195838154872904], [-0.981255069072019], [1.0521099187388827], [1.1240072822724865],
                   [2.0], [1.0260517738498387]]
    a_matrix = np.array([[0, 2, 0], [1, 0, 0], [0, 0, 1]])
    b_matrix = np.array([[1], [1], [0]], dtype=float)

    n_locations = len(step_inputs)
    for idx in range(n_locations):
        a_matrices.append(a_matrix)
        c_vector = np.array(np.matmul(b_matrix, step_inputs[idx]))
        c_vector[2] = 0.1
        print(c_vector)
        c_vectors.append(c_vector)
        max_steps.append(1)

    ref_state = np.array([1, 1, 0])
    ref_simulation = compute_simulation(ref_state, a_matrices, c_vectors, max_steps, 1, disc_dyn=True)
    print(ref_simulation)
    sim_t = np.array(ref_simulation).T
    plt.plot(sim_t[0], sim_t[1], 'b', linestyle='--')
    plt.show()

