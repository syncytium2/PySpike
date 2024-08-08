import numpy as np
import math
import pyspike as spk
import matplotlib.pyplot as plt

def Spike_time_difference_matrix(spike_trains):
    """
    Calculates the spike time difference matrix (STDM) for a given list of spike trains.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :return: The spike time difference matrix.
    :rtype: numpy.ndarray
    """
    num_trains = len(spike_trains)
    indices = np.arange(num_trains)
    assert (indices < num_trains).all() and (indices >= 0).all(),"Invalid index list."
    pairs = [(indices[i], j) for i in range(len(indices)) for j in indices[i+1:]]
    matrix = np.zeros((num_trains,num_trains))
    for i, j in pairs:
        spike_train_order_profile = spk.spike_train_order_profile([spike_trains[i], spike_trains[j]])
        times, e_values = spike_train_order_profile.get_multi_plottable_data([spike_trains[i], spike_trains[j]])
        value = 0
        k = 0
        num_coin = 0
        while k < len(e_values)-1:
            if e_values[k] == -1:
                value += times[k]-times[k+1]
                num_coin += 1
                k += 2
            elif e_values[k] == 1:
                value += times[k+1]-times[k]
                num_coin += 1
                k += 2
            else:
                k += 1
        for k in range(len(times)-1):
            if times[k] == times[k+1]:
                num_coin += 1
        if num_coin == 0:
            matrix[i][j] = 0
            matrix[j][i] = 0
        else:
            matrix[i][j] = value/num_coin
            matrix[j][i] = -value/num_coin
    return matrix

def cost_matrix(spike_trains):
    """
    Calculates the cost matrix for a given list of spike trains.

    .. math::
    \delta^{(n,m)} = \sqrt{\frac{1}{\sum_{i} C^{(n,m)}_i} \sum_{i} C^{(n,m)}_i \left[ \delta^{(n,m)}_i \right]^{2}}

    where 

    .. math::
    \delta^{(n,m)}_i = \left| t_{i}^{(n)} - t_{j^{\prime}}^{(m)} \right| \quad \text{and} \quad j^{\prime} = \arg\min_j(\left| t_{i}^{(n)} - t_{j}^{(m)} \right|)


    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :return: The cost matrix.
    :rtype: numpy.ndarray
    """
    num_trains = len(spike_trains)
    
    indices = np.arange(num_trains)
    assert (indices < num_trains).all() and (indices >= 0).all(),"Invalid index list."
    pairs = [(indices[i], j) for i in range(len(indices)) for j in indices[i+1:]]

    Mat = np.zeros((num_trains, num_trains))
    for i, j in pairs:
        spike_sync_profile = spk.spike_sync_profile([spike_trains[i], spike_trains[j]])
        time, c_prof = spike_sync_profile.get_multi_plottable_data([spike_trains[i], spike_trains[j]])
        c_prof_sum = sum(c_prof)/2
        cost = 0
        if c_prof_sum != 0:
            k = 0
            while k < len(time)-1:
                if c_prof[k] == 1:
                    cost += (time[k]-time[k+1])**2
                    k += 2
                else:
                    k += 1
            cost /= c_prof_sum
        Mat[i][j] = np.sqrt(cost)
        Mat[j][i] = Mat[i][j]
    return Mat

def latency_correction_extrapol(spike_diffs_mat, stop_diagonal=-1):
    """
    Applies latency correction using extrapolation based on the spike time difference matrix.

    :param spike_diffs_mat: The spike time difference matrix.
    :type spike_diffs_mat: numpy.ndarray
    :param stop_diagonal: The diagonal to stop the extrapolation, defaults to -1 (which means using the max value of the first row).
    :type stop_diagonal: int, optional
    :return: The shifts for latency correction, the zero counter, and the normalized zero counter.
    :rtype: tuple

    Example::

            STDM = Spike_time_difference_matrix(spike_trains)
            shifts, zero_counter, norm_zero_counter = latency_correction_extrapol(STDM)
    """
    if stop_diagonal == -1:
        stop_diagonal = np.argmax(np.abs(spike_diffs_mat[0]))+1
    num_trains = spike_diffs_mat.shape[0]
    zero_counter = 0

    for d in range(stop_diagonal + 1, num_trains - 1):
        mask = np.triu(np.ones((num_trains, num_trains)), -d + 1) - np.triu(np.ones((num_trains, num_trains)), d)
        
        for i in range(num_trains - d):
            mask_vec = mask[i, :] * mask[:, i + d]
            
            spike_diffs_mat[i, i + d] = np.sum((spike_diffs_mat[i, :] + spike_diffs_mat[:, i + d]) * mask_vec) / np.sum(mask_vec)
            spike_diffs_mat[i + d, i] = -spike_diffs_mat[i, i + d]
    
    shifts = np.mean(spike_diffs_mat, axis=1)
    
    if stop_diagonal < num_trains - 1:
        norm_zero_counter = zero_counter / ((num_trains - stop_diagonal) * (num_trains - stop_diagonal - 1) / 2)
    else:
        norm_zero_counter = zero_counter
    shifts = shifts - np.mean(shifts)
    return -shifts, zero_counter, norm_zero_counter
    

def latency_correction_intrapol(spike_diffs_mat, stop_diagonal=-1):
    """
    Applies latency correction using interpolation based on the spike time difference matrix.

    :param spike_diffs_mat: The spike time difference matrix.
    :type spike_diffs_mat: numpy.ndarray
    :param stop_diagonal: The diagonal to stop the interpolation, defaults to -1 (which means using the max value of the first row).
    :type stop_diagonal: int, optional
    :return: The shifts for latency correction, the zero counter, and the normalized zero counter.
    :rtype: tuple

    Example::

            STDM = Spike_time_difference_matrix(spike_trains)
            shifts, zero_counter, norm_zero_counter = latency_correction_intrapol(STDM)
    """
    if stop_diagonal == -1:
        stop_diagonal = np.argmax(np.abs(spike_diffs_mat[0]))+1
    num_trains = spike_diffs_mat.shape[0]
    first_diagonal = np.zeros(num_trains - 1)
    zero_counter = 0

    mask = np.triu(np.ones((num_trains, num_trains)), -stop_diagonal) - np.triu(np.ones((num_trains, num_trains)), stop_diagonal + 1)

    for dc in range(1, num_trains):
        fmask_vec = np.where(mask[dc - 1, :] * mask[:, dc])[0]

        if len(fmask_vec) > 0:
            first_diagonal[dc - 1] = np.sum(spike_diffs_mat[dc - 1, fmask_vec] + spike_diffs_mat[fmask_vec, dc].T) / len(fmask_vec)
        else:
            zero_counter += 1

    shifts = np.concatenate(([0], np.cumsum(first_diagonal)))
    norm_zero_counter = zero_counter / (num_trains - 1)
    shifts = shifts - np.mean(shifts)
    return shifts, zero_counter, norm_zero_counter

def sim_ann_latency_correction(spike_trains):
    """
    Performs simulated annealing latency correction on spike trains.

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :returns: A tuple containing:
        - all_shifts: The shifts applied to each spike train.
        - acceptance_prob: The acceptance probability of the simulated annealing process.
        - sa_end_cost: The final cost of the simulated annealing process.
    :rtype: tuple
    """
    try:
        from .generic import Multi_Profile_Matrix
    except ImportError:
        raise ImportError("Error: Could not import Multi_Profile_Matrix from pyspike.generic.")
    try:
        from .spikes import f_all_trains
    except ImportError:
        raise ImportError("Error: Could not import f_all_trains from pyspike.spikes.")

    num_trains = len(spike_trains)
    start_tim = []
    for i in range(num_trains):
        for j in range(len(spike_trains[i])):
            start_tim.append(spike_trains[i][j])
    start_tim.sort()
    start_tim = np.array(start_tim)
    SPIKE_synchro_mat = spk.spike_sync_matrix(spike_trains)
    ssm = Multi_Profile_Matrix(spike_trains, 1)

    sim_ann_temp_fact = 10000
    sim_ann_stop_diagonal = num_trains-1
    iter_unit = 1000000

    ss_matches = []
    for i in range(len(ssm)):
        ss_matches.append(sum(ssm[i])/2)

    all_trains = f_all_trains(spike_trains)[0]
    indices = np.arange(num_trains)
    assert (indices < num_trains).all() and (indices >= 0).all(),"Invalid index list."
    pairs = [(indices[i], j) for i in range(len(indices)) for j in indices[i+1:]]
    indies = np.zeros((num_trains, num_trains-1))

    for trc in range(num_trains):
        indices = [i for i, pair in enumerate(pairs) if trc in pair]
        indies[trc] = indices
    indies = np.sort(indies, axis=1).astype(int)

    num_spikes = []
    for i in range(num_trains):
        num_spikes.append(len(spike_trains[i]))

    all_shifts = np.zeros(num_trains)
    num_pairs = (int)(num_trains*(num_trains-1)/2)
    start_rmse = np.zeros(num_pairs)
    for i in range(num_pairs):
        if ss_matches[i] > 0:
            start_rmse[i]=math.sqrt(np.mean((start_tim[np.logical_and(ssm[i, :], all_trains==pairs[i][1]+1)]-start_tim[np.logical_and(ssm[i, :], all_trains==pairs[i][0]+1)])**2))
    start_rmse[start_rmse < 1e-14] = 0

    sa_mask_mat = np.triu(np.ones((num_trains, num_trains)), -sim_ann_stop_diagonal) - np.triu(np.ones((num_trains, num_trains)), sim_ann_stop_diagonal + 1)
    sa_masks = sa_mask_mat[np.tril(np.ones((num_trains, num_trains)), -1).astype(bool)]
    sa_start_cost = np.mean(start_rmse[sa_masks.astype(bool)])

    old_tim = start_tim.copy()
    old_rmse = start_rmse
    sa_old_cost = sa_start_cost
    sa_costs = sa_start_cost

    keep_range = np.any(np.diag(SPIKE_synchro_mat, k=1) == 0)
    if keep_range:
        diagonal_problem = np.where(np.diag(SPIKE_synchro_mat, 1) == 0)[0]
        first_indies = np.zeros(num_trains, dtype=int)
        last_indies = np.zeros(num_trains, dtype=int)
        
        for trc in range(num_trains):
            if num_spikes[trc] > 0:
                first_indies[trc] = np.where(np.array(all_trains) == trc + 1)[0][0]
                last_indies[trc] = np.where(np.array(all_trains) == trc + 1)[0][-1]
        start_separation = np.max(start_tim[first_indies[first_indies > 0]-1]) - np.min(start_tim[last_indies[last_indies > 0]-1])
        sa_separations = start_separation
    else:
        first_indies = np.zeros(num_trains, dtype=int)
        last_indies = np.zeros(num_trains, dtype=int)

    a = sa_costs
    sa_costs = np.zeros(iter_unit)
    sa_costs[0] = a
    sa_costs[-1] = np.nan

    if keep_range:
        a = sa_separations
        sa_separations = np.zeros(iter_unit)
        sa_separations[0] = a
        sa_separations[-1] = np.nan
        
    min_cost = sa_start_cost
    min_tim = start_tim.copy()
    T = 1
    T_end = T/sim_ann_temp_fact
    alpha = 0.9
    min_iter = 0
    total_iter = 1
    sum_condi = 0
    while T > T_end:
        iterations = 0
        succ_iter = 0
        while (iterations < 100*num_trains) and (succ_iter < 10*num_trains):
            new_tim = old_tim.copy()
            train = np.random.randint(0, num_trains)
            displacement = np.random.randn(1)*sa_old_cost
            for i in range(len(all_trains)):
                if all_trains[i] == train+1:
                    new_tim[i] = old_tim[i].item()+displacement.item()
            new_rmse = old_rmse.copy()

            for i in indies[train]:
                if ss_matches[i] > 0:
                    new_rmse[i]=math.sqrt(np.mean((new_tim[np.logical_and(ssm[i, :], all_trains==pairs[i][1]+1)]-new_tim[np.logical_and(ssm[i, :], all_trains==pairs[i][0]+1)])**2))
                        
            sa_new_cost = np.mean(new_rmse[sa_masks.astype(bool)])
            if keep_range:
                L, M = [], []
                for i in range(num_trains):
                    L.append(new_tim[first_indies[i]])
                    M.append(new_tim[last_indies[i]])
                sa_separation = np.max(L)-np.min(M)
                sa_delta_cost = sa_new_cost+10000*(sa_separation>0)-sa_old_cost
            else:
                sa_delta_cost = sa_new_cost-sa_old_cost

            condi = (sa_delta_cost<0) or (math.exp(-sa_delta_cost/T)>np.random.rand())
            sum_condi = sum_condi+condi
                
            if condi:
                old_tim = new_tim.copy()
                old_rmse = new_rmse.copy() # needed for plotting (and for correct updating!!!)
                sa_old_cost = sa_new_cost
                succ_iter = succ_iter+1
                if sa_new_cost < min_cost:
                    min_iter = total_iter+iterations
                    sa_min_cost = sa_new_cost
                    min_tim = old_tim.copy()

            iterations = iterations+1
            sa_costs[total_iter+iterations-1] = sa_old_cost
            if keep_range:
                if condi:
                    sa_separations[total_iter+iterations-1] = sa_separation
                else:
                    sa_separations[total_iter+iterations-1] = sa_separations[total_iter+iterations-2]
                        
        total_iter += iterations
        if (total_iter%iter_unit) > iter_unit-100*num_trains:
            sa_costs[(math.ceil(total_iter/iter_unit)+1)*iter_unit-1] = np.nan # initialization of next 'iter_unit' iterations
            if keep_range:
                sa_separations[(math.ceil(total_iter/iter_unit)+1)*iter_unit-1] = np.nan # initialization of next 'iter_unit' iterations
        T = T*alpha
        if succ_iter==0:
            break
        
    sa_costs = sa_costs[:total_iter]
    sa_min_cost = np.min(sa_costs)
    
    # statistics and actual shifts
    min_shifts = start_tim-min_tim
    for trc in range(num_trains):
        a = []
        for i in range(len(min_shifts)):
            if all_trains[i] == trc+1:
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


    sa_end_cost = sa_min_cost
    acceptance_prob = sum_condi / total_iter * 100
    all_shifts = all_shifts - np.mean(all_shifts)
    return all_shifts, acceptance_prob, sa_end_cost