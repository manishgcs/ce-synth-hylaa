from hylaa.hybrid_automaton import LinearHybridAutomaton, LinearConstraint, HyperRectangle
from hylaa.star import init_hr_to_star
from hylaa.engine import HylaaEngine
from hylaa.pv_container import PVObject


def define_ha(settings, args):

    step_size = settings.step

    # x' = Ax + Bu + c
    '''make the hybrid automaton and return it'''

    ha = LinearHybridAutomaton()

    a_matrices = args[0]
    t_jumps = list(args[1])
    usafe_arg = args[2]

    n_locations = len(a_matrices)
    n_variables = len(a_matrices[0][0])
    print(n_locations, n_variables)

    ha.variables = []
    for idx in range(n_variables-1):
        x_var_name = "x"+str(idx)
        ha.variables.append(x_var_name)
    ha.variables.append("t")

    locations = []
    for idx in range(n_locations):
        loc_name = 'loc' + str(idx)
        loc = ha.new_mode(loc_name)
        loc.a_matrix = a_matrices[idx]
        c_vector = [0.0] * n_variables
        c_vector[n_variables-1] = 1
        loc.c_vector = c_vector
        if idx == 0:
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = 1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, step_size * t_jumps[idx]))
        elif idx == n_locations - 1:
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = -1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, -step_size * t_jumps[idx-1]))
        else:
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = -1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, -step_size * t_jumps[idx-1]))
            loc_inv_vec = [0.0] * n_variables
            loc_inv_vec[n_variables - 1] = 1.0
            loc.inv_list.append(LinearConstraint(loc_inv_vec, step_size * t_jumps[idx]))
        locations.append(loc)

    for idx in range(n_locations-1):
        trans = ha.new_transition(locations[idx], locations[idx+1])
        trans_vec = [0.0] * n_variables
        trans_vec[n_variables - 1] = -1.0
        trans.condition_list.append(LinearConstraint(trans_vec, -step_size*(t_jumps[idx]+1)))

    error = ha.new_mode('_error')
    error.is_error = True

    usafe_set_constraint_list = []
    if usafe_arg is not None:
        if isinstance(usafe_arg, HyperRectangle):
            usafe_r = usafe_arg
            usafe_star = init_hr_to_star(settings, usafe_r, ha.modes['_error'])
            for constraint in usafe_star.constraint_list:
                usafe_set_constraint_list.append(constraint)
        elif isinstance(usafe_arg, list):
            for idx in range(len(usafe_arg)):
                u_lin_constraint = usafe_arg[idx]
                usafe_set_constraint_list.append(u_lin_constraint)

    for idx in range(n_locations):
        trans_u = ha.new_transition(locations[idx], error)
        for constraint in usafe_set_constraint_list:
            trans_u.condition_list.append(constraint)

    return ha, usafe_set_constraint_list


def define_init_states(ha, init_r):
    '''returns a list of (mode, HyperRectangle)'''
    # Variable ordering: [x, y]
    rv = []
    rv.append((ha.modes['loc0'], init_r))

    return rv


def run_hylaa(settings, init_r, *args):

    'Runs hylaa with the given settings, returning the HylaaResult object.'

    ha, usafe_set_constraint_list = define_ha(settings, args)
    init = define_init_states(ha, init_r)

    engine = HylaaEngine(ha, settings)
    reach_tree = engine.run(init)

    return PVObject(len(ha.variables), usafe_set_constraint_list, reach_tree)

