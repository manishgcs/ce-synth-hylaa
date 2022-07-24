from controlcore.control_utils import get_input, extend_a_b
from controlcore.ha_define_eval import run_hylaa
import numpy as np
from hylaa.hybrid_automaton import HyperRectangle


def find_safe_controller(a_matrix, b_matrix, init_r, usafe_arg, settings):
    a_matrix_ext, b_matrix_ext = extend_a_b(a_matrix, b_matrix)

    R_mult_factor = 0.01

    Q_matrix = np.eye(len(a_matrix[0]), dtype=float)

    u_dim = len(b_matrix[0])
    R_matrix = R_mult_factor * np.eye(u_dim)
    k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)

    a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
    old_pv_object = run_hylaa(settings, init_r, [a_bk_matrix], [int(settings.max_time / settings.step)], usafe_arg)
    lce_object = old_pv_object.compute_longest_ce(control_synth=True)

    if lce_object.get_counterexample() is None:
        print("The system is safe")
        return None, k_matrix
    else:
        last_pv_object = None
        depth_direction = np.eye(len(init_r.dims), dtype=float)
        old_depth_diffs = []
        for idx in range(len(init_r.dims) - 1):
            deepest_ce_1 = old_pv_object.compute_deepest_ce(depth_direction[idx])
            deepest_ce_2 = old_pv_object.compute_deepest_ce(-depth_direction[idx])
            old_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

        t_jumps = lce_object.switching_times
        n_locations = len(t_jumps) + 1
        loc_idx = 1

        Q_mult_factors = np.ones(n_locations)
        Q_matrix = np.eye(len(a_matrix[0]))
        Q_matrices = [Q_matrix]
        k_matrices = [k_matrix]

        for idx in range(len(t_jumps)):
            Q_matrix = np.eye(len(a_matrix[0]))
            Q_matrices.append(Q_matrix)

            R_matrix = R_mult_factor * np.eye(u_dim)
            k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
            k_matrices.append(k_matrix)

        a_bk_matrices = []
        for idx in range(len(k_matrices)):
            k_matrix = k_matrices[idx]
            a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
            a_bk_matrices.append(a_bk_matrix)

        Q_matrix = np.multiply(Q_matrices[loc_idx], old_depth_diffs)
        k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
        a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[1])

        new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_arg)
        usafe_interval = new_pv_object.compute_usafe_interval(loc_idx)

        Q_weigths = []
        lce_object = new_pv_object.compute_longest_ce(control_synth=True)
        if lce_object.get_counterexample() is None:
            Q_weigths = old_depth_diffs
            print("The system is safe")
            print(Q_mult_factors, Q_weigths)
            return t_jumps, k_matrix
        elif (usafe_interval[1] - usafe_interval[0] + 1) < (t_jumps[1] - t_jumps[0]):
            Q_weigths = old_depth_diffs
        else:
            depth_direction = np.identity(len(init_r.dims))
            new_depth_diffs = []
            for idx in range(len(init_r.dims) - 1):
                deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
                deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
                new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

            for idx in range(len(new_depth_diffs)):
                if new_depth_diffs[idx] - old_depth_diffs[idx] < 0:
                    Q_weigths.append(1/old_depth_diffs[idx])
                else:
                    Q_weigths.append(1/old_depth_diffs[idx])

        mult_factors = [10, 0.1]
        for m_factor in mult_factors:
            Q_mult_factor = m_factor
            Q_matrix = m_factor * np.multiply(Q_matrices[1], Q_weigths)
            k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
            a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[loc_idx])

            new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_arg)
            u_interval = new_pv_object.compute_usafe_interval(loc_idx)
            if u_interval[0] == u_interval[1] == -1:
                last_pv_object = new_pv_object
                Q_mult_factors[loc_idx] = Q_mult_factor
                loc_idx = loc_idx + 1
                break
            depth_direction = np.eye(len(init_r.dims))
            new_depth_diffs = []
            for idx in range(len(init_r.dims) - 1):
                deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
                deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
                new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

            if new_depth_diffs < old_depth_diffs:
                Q_mult_factors[loc_idx] = Q_mult_factor
                break

        # Check once if the system is safe, after applying the controller on loc1
        if last_pv_object is not None:
            lce_object = last_pv_object.compute_longest_ce()
            if lce_object.get_counterexample() is None:
                print("The system is safe")
                print(Q_mult_factors, Q_weigths)
                return t_jumps, k_matrices

        Q_mult_factor = Q_mult_factors[1]  # for loc 1
        assert Q_mult_factor == 0.1 or Q_mult_factor == 10
        Q_weigths = np.ones(len(Q_weigths))
        # Q_weigths = [Q_mult_factor * weight for weight in Q_weigths]
        for loc in range(loc_idx, n_locations):
            if last_pv_object is not None:
                u_interval = last_pv_object.compute_usafe_interval(loc)
                if u_interval[0] == u_interval[1] == -1:
                    continue
            else:
                print("Last pv object is None")
            current_loc_safe = False
            while current_loc_safe is False:
                Q_matrix = Q_matrices[loc]
                Q_mult_factors[loc] = Q_mult_factor * Q_mult_factors[loc]
                Q_matrix = Q_mult_factors[loc] * np.multiply(Q_matrix, Q_weigths)
                R_matrix = R_mult_factor * np.eye(u_dim)
                k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
                k_matrices[loc] = k_matrix
                a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
                a_bk_matrices[loc] = a_bk_matrix
                new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_arg)
                u_interval = new_pv_object.compute_usafe_interval(loc_idx)
                if u_interval[0] == u_interval[1] == -1:
                    current_loc_safe = True
                    loc_idx = loc_idx + 1
                    last_pv_object = new_pv_object
        print(Q_mult_factors, Q_weigths)
        return t_jumps, k_matrices


def find_safe_controller_dce(a_matrix, b_matrix, init_r, usafe_arg, settings):
    iter_count = 0

    a_matrix_ext, b_matrix_ext = extend_a_b(a_matrix, b_matrix)

    R_mult_factor = 0.01

    Q_matrix = np.eye(len(a_matrix[0]), dtype=float)

    u_dim = len(b_matrix[0])
    R_matrix = R_mult_factor * np.eye(u_dim)
    k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)

    a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
    old_pv_object = run_hylaa(settings, init_r, [a_bk_matrix], [int(settings.max_time / settings.step)], usafe_arg)
    iter_count = iter_count + 1
    lce_object = old_pv_object.compute_longest_ce(control_synth=True)

    if lce_object.get_counterexample() is None:
        print("The system is safe")
        return None, k_matrix
    else:
        lce = lce_object.get_counterexample()
        lce_init_r = []
        for idx in range(len(lce)):
            temp_tuple = (lce[idx], lce[idx])
            lce_init_r.append(temp_tuple)
        lce_init_r = HyperRectangle(lce_init_r)
        # init_r = lce_init_r
        last_pv_object = None
        depth_direction = np.identity(len(init_r.dims))
        old_depth_diffs = []
        for idx in range(len(init_r.dims) - 1):
            deepest_ce_1 = old_pv_object.compute_deepest_ce(depth_direction[idx])
            deepest_ce_2 = old_pv_object.compute_deepest_ce(-depth_direction[idx])
            old_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

        t_jumps = lce_object.switching_times
        n_locations = len(t_jumps) + 1
        loc_idx = 1

        Q_mult_factors = np.ones(n_locations)
        Q_matrix = np.eye(len(a_matrix[0]))
        Q_matrices = [Q_matrix]
        k_matrices = [k_matrix]

        for idx in range(len(t_jumps)):
            Q_matrix = np.eye(len(a_matrix[0]))
            Q_matrices.append(Q_matrix)

            R_matrix = R_mult_factor * np.eye(u_dim)
            k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
            k_matrices.append(k_matrix)

        a_bk_matrices = []
        for idx in range(len(k_matrices)):
            k_matrix = k_matrices[idx]
            a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
            a_bk_matrices.append(a_bk_matrix)

        # old_depth_diffs = np.ones(len(init_r.dims)-1)
        Q_matrix = np.multiply(Q_matrices[loc_idx], old_depth_diffs)
        k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
        a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[1])

        new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_arg)
        iter_count = iter_count + 1
        usafe_interval = new_pv_object.compute_usafe_interval(loc_idx)

        Q_weigths = []
        lce_object = new_pv_object.compute_longest_ce(control_synth=True)
        if lce_object.get_counterexample() is None:
            Q_weigths = old_depth_diffs
            print("The system is safe")
            print(Q_mult_factors, Q_weigths)
            print(iter_count)
            return t_jumps, k_matrix
        elif (usafe_interval[1] - usafe_interval[0] + 1) < (t_jumps[1] - t_jumps[0]):
            Q_weigths = old_depth_diffs
            last_pv_object = new_pv_object
        else:
            depth_direction = np.identity(len(init_r.dims))
            new_depth_diffs = []
            for idx in range(len(init_r.dims) - 1):
                deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
                deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
                new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

            for idx in range(len(new_depth_diffs)):
                if new_depth_diffs[idx] - old_depth_diffs[idx] < 0:
                    Q_weigths.append(old_depth_diffs[idx])
                else:
                    Q_weigths.append(1/old_depth_diffs[idx])
            last_pv_object = new_pv_object

        # Q_weigths = np.ones(len(Q_weigths))
        # Q_weigths = [0.1*weight for weight in Q_weigths]
        print(Q_weigths)
        mult_factors = [24, 0.05]
        for m_factor in mult_factors:
            Q_mult_factor = m_factor
            Q_matrix = m_factor * np.multiply(Q_matrices[loc_idx], Q_weigths)
            k_matrices[loc_idx] = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
            a_bk_matrices[loc_idx] = a_matrix_ext - np.matmul(b_matrix_ext, k_matrices[loc_idx])

            new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_arg)
            iter_count = iter_count + 1
            u_interval = new_pv_object.compute_usafe_interval(loc_idx)
            if u_interval[0] == u_interval[1] == -1:
                last_pv_object = new_pv_object
                Q_mult_factors[loc_idx] = Q_mult_factor
                loc_idx = loc_idx + 1
                break
            depth_direction = np.identity(len(init_r.dims))
            new_depth_diffs = []
            for idx in range(len(init_r.dims) - 1):
                deepest_ce_1 = new_pv_object.compute_deepest_ce(depth_direction[idx])
                deepest_ce_2 = new_pv_object.compute_deepest_ce(-depth_direction[idx])
                new_depth_diffs.append(abs(deepest_ce_1.ce_depth - deepest_ce_2.ce_depth))

            if new_depth_diffs < old_depth_diffs:
                Q_mult_factors[loc_idx] = Q_mult_factor
                break

        # Check once if the system is safe, after applying the controller on loc1
        if last_pv_object is not None:
            lce_object = last_pv_object.compute_longest_ce()
            if lce_object.get_counterexample() is None:
                Q_matrices[1] = Q_mult_factors[1] * np.multiply(Q_matrices[1], Q_weigths)
                print("The system is safe")
                print(Q_mult_factors, Q_weigths)
                print(Q_matrices)
                print(iter_count)
                return t_jumps, k_matrices

        Q_mult_factor = Q_mult_factors[1]
        assert Q_mult_factor == Q_mult_factors[0] or Q_mult_factor == Q_mult_factors[1]
        prev_Q_weights = [Q_mult_factor*weight for weight in Q_weigths]
        print(Q_weigths, prev_Q_weights)
        # print(prev_Q_weights * prev_Q_weights)
        for loc in range(loc_idx, n_locations):
            if last_pv_object is not None:
                u_interval = last_pv_object.compute_usafe_interval(loc)
                if u_interval[0] == u_interval[1] == -1:
                    continue
            else:
                print("Last pv object is None")
            current_loc_safe = False
            while current_loc_safe is False:
                Q_matrix = Q_matrices[loc]
                # Q_mult_factors[loc] = Q_mult_factor * Q_mult_factors[loc]
                if Q_mult_factor < 1:
                    prev_Q_weights = [0.5 * weight for weight in prev_Q_weights]
                else:
                    prev_Q_weights = [2 * weight for weight in prev_Q_weights]
                Q_matrix = np.multiply(Q_matrix, prev_Q_weights)
                print(Q_matrix)
                # Q_matrices[loc] = Q_matrix
                R_matrix = R_mult_factor * np.eye(u_dim)
                k_matrix = get_input(a_matrix, b_matrix, Q_matrix, R_matrix)
                k_matrices[loc] = k_matrix
                a_bk_matrix = a_matrix_ext - np.matmul(b_matrix_ext, k_matrix)
                a_bk_matrices[loc] = a_bk_matrix
                new_pv_object = run_hylaa(settings, init_r, a_bk_matrices, t_jumps, usafe_arg)
                iter_count = iter_count + 1
                u_interval = new_pv_object.compute_usafe_interval(loc_idx)
                if u_interval[0] == u_interval[1] == -1:
                    current_loc_safe = True
                    Q_matrices[loc_idx] = Q_matrix
                    loc_idx = loc_idx + 1
                    last_pv_object = new_pv_object
                    prev_Q_weights = [Q_mult_factor*weight for weight in Q_weigths]
        print(Q_mult_factors, prev_Q_weights)
        print(Q_matrices)
        print(iter_count)
        return t_jumps, k_matrices
