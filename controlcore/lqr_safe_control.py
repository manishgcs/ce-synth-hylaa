from controlcore.control_utils import get_input, extend_a_b
from controlcore.ha_define_eval import run_hylaa
import numpy as np
from hylaa.hybrid_automaton import HyperRectangle
import scipy.linalg as la


def write_eigens_in_file(a_matrix, loc_idx=1):
    eigen_vals, eigen_vecs = la.eig(a_matrix)
    eigen_file = open('./eigens', 'a')
    eigen_file.write(str(a_matrix))
    eigen_file.write("\n")
    eigen_file.write(str(loc_idx))
    eigen_file.write("\n")
    eigen_file.write(str(eigen_vals))
    eigen_file.write("\n")
    eigen_file.write(str(eigen_vecs))
    eigen_file.write("\n")
    eigen_file.close()


class CntrlObject(object):
    def __init__(self, a_matrix, b_matrix, init_r, usafe_arg, settings):
        self.a_matrix = a_matrix
        self.b_matrix = b_matrix
        self.init_r = init_r
        self.usafe_arg = usafe_arg
        self.settings = settings
        self.t_jumps = []
        self.Q_matrices = []
        self.k_matrices = []
        self.R_matrix = []
        self.iter_count = 0

    def setRMatrix(self, R_matrix):
        self.R_matrix = R_matrix

    def setQMatrices(self, Q_matrices):
        self.Q_matrices = Q_matrices

    def setQMatrix(self, Q_matrix, loc_idx):
        self.Q_matrices[loc_idx] = Q_matrix

    def setKMatrices(self, k_matrices):
        self.k_matrices = k_matrices

    def setKMatrix(self, k_matrix, loc_idx):
        self.k_matrices[loc_idx] = k_matrix

    def loc_safe_controller(self, loc_idx, pv_object, unit_wts_for_Q):
        a_matrix = self.a_matrix
        b_matrix = self.b_matrix
        k_matrices = self.k_matrices
        Q_matrices = self.Q_matrices
        R_matrix = self.R_matrix

        a_matrix_ext, b_matrix_ext = extend_a_b(a_matrix, b_matrix, self.settings.discrete_dyn)
        n_locations = len(self.t_jumps) + 1

        a_bk_matrices = []
        for idx in range(n_locations):
            k_matrix = k_matrices[idx]
            a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
            a_bk_matrices.append(a_bk_matrix)

        last_pv_object = pv_object
        depth_direction = np.identity(len(self.init_r.dims))
        old_depth_diffs = []
        for idx in range(len(self.init_r.dims) - 1):
            deepest_ce_1 = last_pv_object.compute_deepest_ce(depth_direction[idx])
            deepest_ce_2 = last_pv_object.compute_deepest_ce(-depth_direction[idx])
            old_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

        Q_matrix = Q_matrices[loc_idx]
        if unit_wts_for_Q is False:
            Q_matrix = np.multiply(Q_matrices[loc_idx], old_depth_diffs)
            k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix, self.settings.discrete_dyn)
            a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[loc_idx])

        write_eigens_in_file(a_bk_matrices[loc_idx], loc_idx)
        # Q_matrix = Q_matrices[loc_idx]

        print(old_depth_diffs)
        new_pv_object = run_hylaa(self.settings, self.init_r, a_bk_matrices, self.t_jumps, self.usafe_arg)
        self.iter_count = self.iter_count + 1
        usafe_interval = new_pv_object.compute_usafe_interval(loc_idx)

        Q_weigths = []
        lce_object = new_pv_object.compute_longest_ce(control_synth=True)
        if lce_object.get_counterexample() is None:
            self.setQMatrix(Q_matrix, loc_idx)
            self.setKMatrices(k_matrices)
            print("The system is safe")
            return
        elif len(self.t_jumps) > loc_idx and (usafe_interval[1] - usafe_interval[0] + 1) < (self.t_jumps[loc_idx] - self.t_jumps[loc_idx-1]):
            Q_weigths = old_depth_diffs
            last_pv_object = new_pv_object
        else:
            depth_direction = np.identity(len(self.init_r.dims))
            new_depth_diffs = []
            for idx in range(len(self.init_r.dims) - 1):
                deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
                deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
                new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

            for idx in range(len(new_depth_diffs)):
                if new_depth_diffs[idx] - old_depth_diffs[idx] < 0:
                    Q_weigths.append(new_depth_diffs[idx])
                else:
                    Q_weigths.append(old_depth_diffs[idx])
            last_pv_object = new_pv_object

        if unit_wts_for_Q is True:
            Q_weigths = np.ones(len(Q_weigths))

        print(Q_weigths)
        # Q_weigths = [0.1*weight for weight in Q_weigths]

        # This block is just to find the multiplication factor. Depth diffs are not being used afterwards
        print(Q_weigths)
        # mult_factors = [50, 0.02]  ## Use this for ACC new test
        mult_factors = [25, 0.04]
        Q_mult_factor = mult_factors[0]
        for m_factor in mult_factors:
            temp_Q_weights = m_factor * np.asarray(Q_weigths)
            # temp_Q_weights = [weight if weight <= 25.0 else 25.0 for weight in temp_Q_weights]
            Q_matrix = np.multiply(Q_matrices[loc_idx], temp_Q_weights)
            k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix, self.settings.discrete_dyn)
            a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[loc_idx])

            write_eigens_in_file(a_bk_matrices[loc_idx], loc_idx)

            new_pv_object = run_hylaa(self.settings, self.init_r, a_bk_matrices, self.t_jumps, self.usafe_arg)
            self.iter_count = self.iter_count + 1
            u_interval = new_pv_object.compute_usafe_interval(loc_idx)
            if u_interval[0] == u_interval[1] == -1:
                last_pv_object = new_pv_object
                Q_mult_factor = m_factor
                break
            depth_direction = np.identity(len(self.init_r.dims))
            new_depth_diffs = []
            for idx in range(len(self.init_r.dims) - 1):
                deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
                deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
                new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

            if new_depth_diffs < old_depth_diffs:
                Q_mult_factor = m_factor
                old_depth_diffs = new_depth_diffs

        print("********** The mult factor is {}***********".format(Q_mult_factor))
        # Check once if the system is safe, after applying the controller on current loc
        if last_pv_object is not None:
            lce_object = last_pv_object.compute_longest_ce()
            if lce_object.get_counterexample() is None:
                self.setQMatrix(Q_matrix, loc_idx)
                self.setKMatrices(k_matrices)
                print("The system is safe")
                return None

        assert Q_mult_factor == mult_factors[0] or Q_mult_factor == mult_factors[1]
        prev_Q_weights = [Q_mult_factor * weight for weight in Q_weigths]
        print(Q_weigths, prev_Q_weights)
        current_loc_safe = False
        while current_loc_safe is False:
            # Q_mult_factors[loc] = Q_mult_factor * Q_mult_factors[loc]
            if Q_mult_factor < 1:
                prev_Q_weights = [0.75 * weight for weight in prev_Q_weights]
                # prev_Q_weights = [weight if weight >= 0.03 else 0.03 for weight in prev_Q_weights]
            else:
                prev_Q_weights = [1.5 * weight for weight in prev_Q_weights]
                # prev_Q_weights = [weight if weight <= 57.0 else 57.0 for weight in prev_Q_weights]
            Q_matrix = np.multiply(Q_matrices[loc_idx], prev_Q_weights)
            print(Q_matrix)
            # Q_matrices[loc] = Q_matrix
            k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix, self.settings.discrete_dyn)
            k_matrices[loc_idx] = k_matrix
            a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
            a_bk_matrices[loc_idx] = a_bk_matrix

            write_eigens_in_file(a_bk_matrices[loc_idx], loc_idx)

            new_pv_object = run_hylaa(self.settings, self.init_r, a_bk_matrices, self.t_jumps, self.usafe_arg)
            self.iter_count = self.iter_count + 1
            u_interval = new_pv_object.compute_usafe_interval(loc_idx)
            if u_interval[0] == u_interval[1] == -1:
                current_loc_safe = True
                Q_matrices[loc_idx] = Q_matrix
                last_pv_object = new_pv_object

        # Check once if the system is safe, after applying the controller on current loc
        if last_pv_object is not None:
            print(Q_matrix)
            self.setQMatrix(Q_matrix, loc_idx)
            self.setKMatrices(k_matrices)
            lce_object = last_pv_object.compute_longest_ce()
            if lce_object.get_counterexample() is None:
                print("The system is safe")
                return None
            else:
                return last_pv_object

    def find_safe_controller(self, unit_wts_for_Q=False):

        a_matrix_ext, b_matrix_ext = extend_a_b(self.a_matrix, self.b_matrix)
        print(self.a_matrix)
        print(self.b_matrix)

        # R_mult_factor = 0.1  # for quadcopter1

        # R_mult_factor = 1  # For ACC

        R_mult_factor = 0.002  # for inv_pend3
        # R_mult_factor = 0.01  # other benchmarks

        Q_matrix = np.eye(len(self.a_matrix[0]), dtype=float)

        u_dim = len(self.b_matrix[0])
        R_matrix = R_mult_factor * np.eye(u_dim)
        k_matrix = get_input(self.a_matrix, self.b_matrix, Q_matrix, R_matrix)

        a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
        old_pv_object = run_hylaa(self.settings, self.init_r, [a_bk_matrix], [int(self.settings.max_time / self.settings.step)], self.usafe_arg)
        self.iter_count = self.iter_count + 1
        lce_object = old_pv_object.compute_longest_ce(control_synth=True)
        # lce_object = old_pv_object.compute_robust_ce(control_synth=True)
        write_eigens_in_file(a_bk_matrix)

        if lce_object.get_counterexample() is None:
            print("The system is safe")
            self.setQMatrices([Q_matrix])
            self.setKMatrices([k_matrix])
            self.setRMatrix(R_matrix)
            return
        else:
            t_jumps = lce_object.switching_times
            self.t_jumps = t_jumps
            loc_idx = 1

            Q_matrix = np.eye(len(self.a_matrix[0]))
            Q_matrices = [Q_matrix]
            k_matrices = [k_matrix]

            for idx in range(len(t_jumps)):
                Q_matrix = np.eye(len(self.a_matrix[0]))
                Q_matrices.append(Q_matrix)

                k_matrix = get_input(self.a_matrix, self.b_matrix, Q_matrix, R_matrix)
                k_matrices.append(k_matrix)

            R_matrix = R_mult_factor * np.eye(u_dim)
            self.setRMatrix(R_matrix)
            self.setQMatrices(Q_matrices)
            self.setKMatrices(k_matrices)

            while True:
                print("****************Computing control for location {} ****************".format(loc_idx))
                new_pv_object = self.loc_safe_controller(loc_idx, old_pv_object, unit_wts_for_Q)
                if new_pv_object is None:
                    print("Heyaaaaaaaaaa***************** The iter count is {}".format(self.iter_count))
                    break
                else:
                    old_pv_object = new_pv_object
                    loc_idx = loc_idx + 1
            return
