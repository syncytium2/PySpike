"""

Generic functions to compute multi-variate profiles and distance matrices.

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import division
from .isi_lengths import default_thresh
from .spikes import reconcile_spike_trains, reconcile_spike_trains_bi, f_all_trains
import numpy as np

def resolve_keywords(**kwargs):
    """
    Resolve keywords from the given dictionary of keyword arguments.

    Args:
        kwargs (dict): Dictionary of keyword arguments.

    Returns:
        tuple: A tuple containing:
            - MRTS (float): Minimum Relevant Time Scale. Default is 0.
            - RI (bool): Rate Independent Adaptive distance. Default is False.
    """
    if 'MRTS' in kwargs:
        MRTS = kwargs['MRTS']
    else:
        MRTS = 0.  # default
    if 'RI' in kwargs:
        RI = kwargs['RI']
    else:
        RI = False  # default
    return MRTS, RI


############################################################
# _generic_profile_multi
############################################################
def _generic_profile_multi(spike_trains, pair_distance_func, indices=None, **kwargs):
    """ 
    Computes the multi-variate distance for a set of spike trains using the
    `pair_distance_func` to compute pair-wise distances. This function calculates the
    average distance of all pairs of spike trains:
    :math:`S(t) = \frac{2}{N(N-1)} \sum_{<i,j>} S_{i,j}`,
    where the sum goes over all pairs `<i,j>`.

    This is an internal implementation detail. Use `isi_profile_multi` or `spike_profile_multi` instead.

    :param spike_trains: List of spike trains.
    :type spike_trains: list

    :param pair_distance_func: Function computing the distance between two spike trains.
    :type pair_distance_func: function

    :param indices: List of indices defining which spike trains to use. If None, all given spike trains are used.
    :type indices: list, optional

    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :keyword bool Reconcile: Whether to reconcile spike trains. Default is True.

    :returns: 
        - Averaged multi-variate distance of all pairs.
    :rtype: float

    :raises ValueError: If there is an issue with resolving the keywords.
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
        kwargs['Reconcile'] = False

    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        kwargs['MRTS'] = default_thresh(spike_trains)

    def divide_and_conquer(pairs1, pairs2):
        """ recursive calls by splitting the two lists in half.
        """
        L1 = len(pairs1)
        if L1 > 1:
            dist_prof1 = divide_and_conquer(pairs1[:L1//2],
                                            pairs1[L1//2:])
        else:
            dist_prof1 = pair_distance_func(spike_trains[pairs1[0][0]],
                                            spike_trains[pairs1[0][1]],
                                            **kwargs)
        L2 = len(pairs2)
        if L2 > 1:
            dist_prof2 = divide_and_conquer(pairs2[:L2//2],
                                            pairs2[L2//2:])
        else:
            dist_prof2 = pair_distance_func(spike_trains[pairs2[0][0]],
                                            spike_trains[pairs2[0][1]], 
                                            **kwargs)
        dist_prof1.add(dist_prof2)
        return dist_prof1

    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]
    
    L = len(pairs)
    if L > 1:
        # recursive iteration through the list of pairs to get average profile
        avrg_dist = divide_and_conquer(pairs[:len(pairs)//2],
                                       pairs[len(pairs)//2:])
    else:
        avrg_dist = pair_distance_func(spike_trains[pairs[0][0]],
                                       spike_trains[pairs[0][1]], 
                                       **kwargs)
    return avrg_dist, L


############################################################
# _generic_distance_multi
############################################################
def _generic_distance_multi(spike_trains, pair_distance_func,
                            indices=None, interval=None, **kwargs):
    """ 
    Computes the multi-variate distance for a set of spike trains using the
    `pair_distance_func` to compute pair-wise distances. Specifically, it computes the
    average distance of all pairs of spike trains:
    :math:`S(t) = \frac{2}{N(N-1)} \sum_{<i,j>} D_{i,j}`,
    where the sum goes over all pairs `<i,j>`.

    This is an internal implementation detail. Use `isi_distance_multi` or `spike_distance_multi` instead.

    :param spike_trains: List of spike trains.
    :type spike_trains: list

    :param pair_distance_func: Function computing the distance between two spike trains.
    :type pair_distance_func: function

    :param indices: List of indices defining which spike trains to use. If None, all given spike trains are used.
    :type indices: list, optional

    :param interval: Interval over which to compute the distance.
    :type interval: optional

    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :keyword bool Reconcile: Whether to reconcile spike trains. Default is True.

    :returns: 
        - Averaged multi-variate distance of all pairs.
    :rtype: float

    :raises ValueError: If there is an issue with resolving the keywords.
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
        kwargs['Reconcile'] = False

    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        kwargs['MRTS'] = default_thresh(spike_trains)
    
    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(indices[i], j) for i in range(len(indices))
             for j in indices[i+1:]]

    avrg_dist = 0.0
    for (i, j) in pairs:
        one_dist = pair_distance_func(spike_trains[i], spike_trains[j],
                                        interval, **kwargs)
        avrg_dist += one_dist

    return avrg_dist/len(pairs)


############################################################
# generic_distance_matrix
############################################################
def _generic_distance_matrix(spike_trains, dist_function,
                             indices=None, interval=None, **kwargs):
    """ 
    Computes the time-averaged distance of all pairs of spike trains.

    This is an internal implementation detail. Use `isi_distance_matrix` or `spike_distance_matrix` instead.

    :param spike_trains: List of spike trains.
    :type spike_trains: list

    :param dist_function: Function computing the distance between two spike trains.
    :type dist_function: function

    :param indices: List of indices defining which spike trains to use. If None, all given spike trains are used.
    :type indices: list, optional

    :param interval: Interval over which to compute the distance.
    :type interval: optional

    :param kwargs: Additional keyword arguments.
    :type kwargs: dict

    :keyword bool Reconcile: Whether to reconcile spike trains. Default is True.

    :returns: 
        - A 2D array of size len(indices) x len(indices) containing the average pair-wise distances.
    :rtype: numpy.ndarray

    :raises ValueError: If there is an issue with resolving the keywords.
    """
    if kwargs.get('Reconcile', True):
        spike_trains = reconcile_spike_trains(spike_trains)
        kwargs['Reconcile'] = False
        
    MRTS, RI = resolve_keywords(**kwargs)
    if isinstance(MRTS, str):
        kwargs['MRTS'] = default_thresh(spike_trains)

    if indices is None:
        indices = np.arange(len(spike_trains))
    indices = np.array(indices)
    # check validity of indices
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(), \
        "Invalid index list."
    # generate a list of possible index pairs
    pairs = [(i, j) for i in range(len(indices))
             for j in range(i+1, len(indices))]

    distance_matrix = np.zeros((len(indices), len(indices)))
    for i, j in pairs:
        d = dist_function(spike_trains[indices[i]], spike_trains[indices[j]],
                          interval, **kwargs)
        distance_matrix[i, j] = d
        distance_matrix[j, i] = d
    return distance_matrix

def Multi_Profile_Matrix(spike_trains, variable):
    """
    Computes and returns a matrix contening the profile values of spike trains.
    The returned matrix size is (number of pairs * number of total spikes)

    :param spike_trains: List of spike trains.
    :type spike_trains: List of :class:`pyspike.SpikeTrain`
    :param variable: The type of data represented by the profile:
                     1 for Spike-Synchro,
                     2 for Spike Order,
                     3 for Spike train order.
    :type variable: int
    :returns: A matrix containing the profile values for each pair of spike trains.
    :rtype: numpy.ndarray
    """

    if variable == 1:
        try:
            from .spike_sync import spike_sync_profile as prof
        except ImportError:
            raise ImportError("Error: Could not import spike_sync_profile from pyspike.spike_sync.")
    elif variable == 2:
        try:
            from .spike_order import spike_order_profile as prof
        except ImportError:
            raise ImportError("Error: Could not import spike_order_profile from pyspike.spike_order.")
    elif variable == 3:
        try:
            from .spike_order import spike_train_order_profile as prof
        except ImportError:
            raise ImportError("Error: Could not import spike_train_order_profile from pyspike.spike_order.")
    else:
        raise ValueError("Error: variable must be 1, 2, or 3.")
    all_trains = f_all_trains(spike_trains)[0]
    indices = np.arange(len(spike_trains))
    assert (indices < len(spike_trains)).all() and (indices >= 0).all(),"Invalid index list."
    pairs = [(indices[i], j) for i in range(len(indices)) for j in indices[i+1:]]
    num_pairs = len(pairs)
    num_spikes = len(all_trains)
    Mat = np.zeros((num_pairs, num_spikes))
    
    pairscount = 0
    for i, j in pairs:
        bi_spike_trains = [spike_trains[i], spike_trains[j]]
        if variable == 2:
            sto_prof_bi = prof(bi_spike_trains)
        else:
            sto_prof_bi = prof(bi_spike_trains).get_multi_plottable_data(bi_spike_trains)[1]
        sto_prof_bi_count = 0
        for k in range(num_spikes):
            if all_trains[k] == i+1 or all_trains[k] == j+1:
                Mat[pairscount][k] = sto_prof_bi[sto_prof_bi_count]
                sto_prof_bi_count += 1
        pairscount += 1
    return Mat
