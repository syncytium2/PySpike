#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
cython_simulated_annealing.pyx

cython implementation of a simulated annealing algorithm to find the optimal
spike train order

Note: using cython memoryviews (e.g. double[:]) instead of ndarray objects
improves the performance of spike_distance by a factor of 10!

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

"""
To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_simulated_annealing.pyx

which gives:

  cython_simulated_annealing.html

"""

import numpy as np
cimport numpy as np
import pyspike as spk
cimport pyspike as spk

from libc.math cimport exp
from libc.math cimport fmod
from libc.stdlib cimport rand
from libc.stdlib cimport RAND_MAX

#DTYPE = float
#ctypedef np.float_t DTYPE_t

def sim_ann_cython(double[:, :] D, double T_start, double T_end, double alpha):

    cdef long N = len(D)
    cdef double A = np.sum(np.triu(D, 0))
    cdef long[:] p = np.arange(N)
    cdef double T = T_start
    cdef long iterations
    cdef long succ_iter
    cdef long total_iter = 0
    cdef double delta_A
    cdef long ind1
    cdef long ind2

    while T > T_end:
        iterations = 0
        succ_iter = 0
        # equilibrate for 100*N steps or 10*N successful steps
        while iterations < 100*N and succ_iter < 10*N:
            # exchange two rows and cols
            # ind1 = np.random.randint(N-1)
            ind1 = rand() % (N-1)
            if ind1 < N-1:
                ind2 = ind1+1
            else:  # this can never happen!
                ind2 = 0
            delta_A = -2*D[p[ind1], p[ind2]]
            if delta_A > 0.0 or exp(delta_A/T) > ((1.0*rand()) / RAND_MAX):
                # swap indices
                p[ind1], p[ind2] = p[ind2], p[ind1]
                A += delta_A
                succ_iter += 1
            iterations += 1
        total_iter += iterations
        T *= alpha   # cool down
        if succ_iter == 0:
            # no successful step -> we believe we have converged
            break

    return p, A, total_iter

def sim_ann_latency_correction(spike_trains):
    try:
        from pyspike.generic import Multi_Profile_Matrix
    except ImportError:
        raise ImportError("Error: Could not import Multi_Profile_Matrix from pyspike.generic.")
    try:
        from pyspike.spikes import f_all_trains
    except ImportError:
        raise ImportError("Error: Could not import f_all_trains from pyspike.spikes.")

    cdef int num_trains = len(spike_trains)
    cdef int i, j, trc, iterations, succ_iter, total_iter, num_pairs
    cdef list start_tim = []
    cdef np.ndarray[np.float64_t, ndim=1] start_tim_array
    cdef np.ndarray[np.float64_t, ndim=2] SPIKE_synchro_mat
    cdef np.ndarray[np.float64_t, ndim=2] ssm
    cdef double sim_ann_temp_fact = 10000
    cdef int sim_ann_stop_diagonal = num_trains - 1
    cdef int iter_unit = 1000000

    # Collect all spike times and sort them
    for i in range(num_trains):
        for j in range(len(spike_trains[i])):
            start_tim.append(spike_trains[i][j])
    start_tim.sort()
    start_tim_array = np.array(start_tim, dtype=np.float64)
    SPIKE_synchro_mat = spk.spike_sync_matrix(spike_trains)
    ssm = Multi_Profile_Matrix(spike_trains, 1)

    cdef list ss_matches = []
    for i in range(len(ssm)):
        ss_matches.append(sum(ssm[i]) / 2)

    cdef np.ndarray[np.int32_t, ndim=1] all_trains = f_all_trains(spike_trains)[0]
    cdef np.ndarray[np.int32_t, ndim=1] indices = np.arange(num_trains, dtype=np.int32)
    assert (indices < num_trains).all() and (indices >= 0).all(), "Invalid index list."
    cdef list pairs = [(indices[i], j) for i in range(len(indices)) for j in indices[i+1:]]
    cdef np.ndarray[np.int32_t, ndim=2] indies = np.zeros((num_trains, num_trains - 1), dtype=np.int32)

    for trc in range(num_trains):
        indices = np.array([i for i, pair in enumerate(pairs) if trc in pair])
        indies[trc] = indices
    indies = np.sort(indies, axis=1).astype(np.int32)

    cdef list num_spikes = [len(spike_trains[i]) for i in range(num_trains)]
    cdef np.ndarray[np.float64_t, ndim=1] all_shifts = np.zeros(num_trains, dtype=np.float64)
    num_pairs = int(num_trains * (num_trains - 1) / 2)
    cdef np.ndarray[np.float64_t, ndim=1] start_rmse = np.zeros(num_pairs, dtype=np.float64)

    for i in range(num_pairs):
        if ss_matches[i] > 0:
            start_rmse[i] = np.sqrt(np.mean((start_tim_array[np.logical_and(ssm[i, :], all_trains == pairs[i][1] + 1)] - start_tim_array[np.logical_and(ssm[i, :], all_trains == pairs[i][0] + 1)]) ** 2))
    start_rmse[start_rmse < 1e-14] = 0

    cdef np.ndarray[np.int64_t, ndim=2] sa_mask_mat = np.triu(np.ones((num_trains, num_trains), dtype=np.int64), -sim_ann_stop_diagonal) - np.triu(np.ones((num_trains, num_trains), dtype=np.int64), sim_ann_stop_diagonal + 1)
    cdef np.ndarray[np.int64_t, ndim=1] sa_masks
    sa_masks = sa_mask_mat[np.tril(np.ones((num_trains, num_trains), dtype=np.int64), -1).astype(np.bool_)]
    cdef double sa_start_cost = np.mean(start_rmse[sa_masks.astype(np.bool_)])

    cdef np.ndarray[np.float64_t, ndim=1] old_tim = start_tim_array.copy()
    cdef np.ndarray[np.float64_t, ndim=1] old_rmse = start_rmse
    cdef double sa_old_cost = sa_start_cost
    cdef np.ndarray[np.float64_t, ndim=1] sa_costs = np.zeros(iter_unit, dtype=np.float64)
    sa_costs[0] = sa_start_cost
    cdef int keep_range = np.any(np.diag(SPIKE_synchro_mat, k=1) == 0)
    cdef np.ndarray[np.int32_t, ndim=1] diagonal_problem, first_indies, last_indies
    cdef np.ndarray[np.int32_t, ndim=2] inter

    # Ensure sa_separations is initialized regardless of the keep_range condition
    sa_separations = np.zeros(iter_unit, dtype=np.float64)
    sa_separations[iter_unit-1] = np.nan

    if keep_range:
        diagonal_problem = np.where(np.diag(SPIKE_synchro_mat, 1) == 0)[0]
        first_indies = np.zeros(num_trains, dtype=np.int32)
        last_indies = np.zeros(num_trains, dtype=np.int32)
        
        for trc in range(num_trains):
            if num_spikes[trc] > 0:
                first_indies[trc] = np.where(np.array(all_trains) == trc + 1)[0][0]
                inter = np.where(np.array(all_trains) == trc + 1)[0]
                last_indies[trc] = inter[len(inter)]
        start_separation = np.max(start_tim_array[first_indies[first_indies > 0] - 1]) - np.min(start_tim_array[last_indies[last_indies > 0] - 1])
        sa_separations[0] = start_separation

    sa_costs = np.zeros(iter_unit, dtype=np.float64)
    sa_costs[0] = sa_start_cost
    sa_costs[iter_unit-1] = np.nan

    cdef double min_cost = sa_start_cost
    cdef np.ndarray[np.float64_t, ndim=1] min_tim = start_tim_array.copy()
    cdef double T = 1
    cdef double T_end = T / sim_ann_temp_fact
    cdef double alpha = 0.9
    cdef int min_iter = 0
    total_iter = 1
    cdef int sum_condi = 0
    cdef double displacement, sa_new_cost, sa_delta_cost, sa_separation
    cdef int train

    while T > T_end:
        iterations = 0
        succ_iter = 0
        while (iterations < 100 * num_trains) and (succ_iter < 10 * num_trains):
            new_tim = old_tim.copy()
            train = np.random.randint(0, num_trains)
            displacement = np.random.randn(1) * sa_old_cost
            for i in range(len(all_trains)):
                if all_trains[i] == train + 1:
                    new_tim[i] = old_tim[i] + displacement
            new_rmse = old_rmse.copy()

            for i in indies[train]:
                if ss_matches[i] > 0:
                    new_rmse[i] = np.sqrt(np.mean((new_tim[np.logical_and(ssm[i, :], all_trains == pairs[i][1] + 1)] - new_tim[np.logical_and(ssm[i, :], all_trains == pairs[i][0] + 1)]) ** 2))
                        
            sa_new_cost = np.mean(new_rmse[sa_masks.astype(np.bool_)])
            if keep_range:
                L, M = [], []
                for i in range(num_trains):
                    L.append(new_tim[first_indies[i]])
                    M.append(new_tim[last_indies[i]])
                sa_separation = np.max(L) - np.min(M)
                sa_delta_cost = sa_new_cost + 10000 * (sa_separation > 0) - sa_old_cost
            else:
                sa_delta_cost = sa_new_cost - sa_old_cost

            condi = (sa_delta_cost < 0) or (exp(-sa_delta_cost / T) > np.random.rand())
            sum_condi = sum_condi + condi
                
            if condi:
                old_tim = new_tim.copy()
                old_rmse = new_rmse.copy()  # needed for plotting (and for correct updating!!!)
                sa_old_cost = sa_new_cost
                succ_iter += 1
                if sa_new_cost < min_cost:
                    min_cost = sa_new_cost
                    min_tim = new_tim.copy()
                    min_iter = total_iter
            iterations += 1
            total_iter += 1
            if total_iter % iter_unit == 0:
                sa_costs = np.append(sa_costs, np.zeros(iter_unit, dtype=np.float64))
                if keep_range:
                    sa_separations = np.append(sa_separations, np.zeros(iter_unit, dtype=np.float64))
                sa_costs[total_iter] = sa_old_cost
                if keep_range:
                    sa_separations[total_iter] = sa_separation
        T *= alpha

    sa_costs = sa_costs[:total_iter]
    sa_min_cost = np.min(sa_costs)
    
    # statistics and actual shifts
    min_shifts = start_tim - min_tim
    for trc in range(num_trains):
        a = []
        for i in range(len(min_shifts)):
            if all_trains[i] == trc + 1:
                a.append(min_shifts[i])
        all_shifts[trc] = np.mean(a)

    if np.any(np.isnan(all_shifts)):
        isnans = np.where(np.isnan(all_shifts))[0]
        indies = np.where(~np.isnan(all_shifts))[0]
        if indies.size > 0:
            values = all_shifts[indies]
            all_shifts[isnans] = np.interp(isnans, indies, values)
        else:
            # Handle the case where there are no non-NaN values to interpolate from
            # You can decide to set them to a default value, raise an error, or handle it another way
            all_shifts[isnans] = 0  # For example, set them to 0, or use another default value

    cdef float sa_end_cost = sa_min_cost
    cdef float acceptance_prob = sum_condi / total_iter * 100
    all_shifts = all_shifts - np.mean(all_shifts)

    return all_shifts, acceptance_prob, sa_end_cost
