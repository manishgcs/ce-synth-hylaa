
import numpy as np
from hylaa.glpk_interface import LpInstance
from hylaa.hybrid_automaton import LinearConstraint

class CounterExample(object):
    def __init__(self, init_point, basis_predicates, time_steps, modes, num_dims):
        self.preimage_point = init_point
        self.usafe_basis_predicates = []
        self.error_time_steps = []
        self.error_star_modes = []
        self.num_dims = num_dims

        error_star_modes = []
        prev_mode_name = None
        usafe_basis_predicates = []
        error_time_steps = []
        for index in xrange(len(modes)):
            if (modes[index].name is not prev_mode_name):
                error_star_modes.append(modes[index].name)
                prev_mode_name = modes[index].name
                if (len(usafe_basis_predicates) > 0):
                    self.usafe_basis_predicates.append(usafe_basis_predicates)
                    self.error_time_steps.append(error_time_steps)
                    usafe_basis_predicates = []
                    error_time_steps = []
            usafe_basis_predicates.append(basis_predicates[index])
            error_time_steps.append(time_steps[index])

        self.usafe_basis_predicates.append(usafe_basis_predicates)
        self.error_time_steps.append(error_time_steps)
        self.error_star_modes = modes

        #self.freeze_attrs()

    ## For each init_point computed above, ce_vector is a sequence of 0's and 1's of length = len(time_steps).
    ## 1(0) signifies whether the simulation from this init_point intersects (or not) with the unsafe set at
    ## that particular time step
    def compute_ce_vector(self, simulation, usafe_set_constraint_list, direction=None, sim_start_time = 0, current_mode_idx = 0):
        usafe_points = []
        ce_vector = []
        for time in self.error_time_steps[current_mode_idx]:
           point = simulation[int(time) - sim_start_time]
           usafe_lpi = LpInstance(self.num_dims, self.num_dims)
           identity_matrix = np.identity(self.num_dims)
           usafe_lpi.update_basis_matrix(np.identity(self.num_dims))
           for dim in range(identity_matrix.ndim):
               lc = LinearConstraint(identity_matrix[dim], point[dim])
               usafe_lpi.add_basis_constraint(lc.vector, lc.value)
               lc = LinearConstraint(-1 * identity_matrix[dim], -point[dim])
               usafe_lpi.add_basis_constraint(lc.vector, lc.value)
           for constraints in usafe_set_constraint_list:
               usafe_lpi.add_basis_constraint(constraints.vector, constraints.value)

           direction = np.zeros(self.num_dims)
           usafe_point = np.zeros(self.num_dims)
           is_feasible = usafe_lpi.minimize(direction, usafe_point, error_if_infeasible=False)
           usafe_points.append(usafe_point)
           if is_feasible:
              ce_vector.append(1)
           else:
              ce_vector.append(0)
        return ce_vector

    ## Computes longest subsequence given a ce_vector
    def compute_longest_sequence(self, ce_vector, init_star, direction, current_mode_idx = 0):

        usafe_lpi = LpInstance(init_star.num_dims, init_star.num_dims)
        length = 0

        # First value is the length of the subsequence
        # Second and third values are the indices of the subsequence in the ce_vector
        time_step_indices = [0, 0, 0]
        start_index = 0
        end_index = 0
        index = [0, 0]
        max_len = 0
        current_index = 0
        while (current_index < len(ce_vector)):
            if ce_vector[current_index] == 1:
               if length == 0:
                   usafe_lpi = LpInstance(init_star.num_dims, init_star.num_dims)
                   usafe_lpi.update_basis_matrix(init_star.basis_matrix)
                   for lc in init_star.constraint_list:
                       usafe_lpi.add_basis_constraint(lc.vector, lc.value)
                   start_index = current_index

               usafe_basis_predicates = self.usafe_basis_predicates[current_mode_idx][current_index]
               result = np.zeros(init_star.num_dims)
               for usafe_basis_predicate in usafe_basis_predicates:
                   usafe_lpi.add_basis_constraint(usafe_basis_predicate.vector, usafe_basis_predicate.value)

               is_feasible = usafe_lpi.minimize(direction, result, error_if_infeasible=False)
               if is_feasible:
                   length = length + 1
                   end_index = current_index
               else:
                   current_index = start_index + 1
                   length = 0

               if max_len < length:
                   max_len = length
                   index[0] = start_index
                   index[1] = end_index
            else:
               length = 0
            current_index = current_index + 1

        time_step_indices[1] = self.error_time_steps[current_mode_idx][index[0]]
        time_step_indices[2] = self.error_time_steps[current_mode_idx][index[1]]
        time_step_indices[0] = time_step_indices[2] - time_step_indices[1] + 1
        return time_step_indices

