import numpy as np
import pytest
import pyspike as spk
from numpy.testing import assert_equal, assert_almost_equal, assert_array_equal, assert_array_almost_equal
from pyspike.latency_correction import Spike_time_difference_matrix, latency_correction_extrapol, latency_correction_intrapol, sim_ann_latency_correction, cost_matrix
from pyspike.spikes import create_synfire


def test_latency_correction():

    tmin=0
    tmax = 100 
    num_trains = 5
    num_synfire_events = 4
    num_inverse_events = 0
    overlap = 0.4
    shuffle = 0
    jitter = 0
    complete = 1
    background = 1-complete
    order = 0
    plot = 0
    spike_trains = create_synfire(tmin, tmax, num_trains, num_synfire_events, num_inverse_events, overlap, shuffle, jitter, complete, background, order, plot)[0]

    STDM1_result = Spike_time_difference_matrix(spike_trains)
    result_expected = [np.array([0, 2.9412, 5.8824, 8.8235, 11.7647]), np.array([-2.9412, 0, 2.9412, 5.8824, 8.8235]), np.array([-5.8824, -2.9412, 0, 2.9412, 5.8824]), np.array([-8.8235, -5.8824, -2.9412, 0, 2.9412]), np.array([-11.7647 , -8.8235, -5.8824, -2.9412,0])]
    assert_array_almost_equal(STDM1_result, result_expected, decimal=4)

    result = latency_correction_extrapol(STDM1_result)[0]
    result_expected = np.array([-5.8824, -2.9412,  0.0000,  2.9412,  5.8824])
    assert_array_almost_equal(result, result_expected, decimal=4)

    result = latency_correction_intrapol(STDM1_result)[0]
    result_expected = np.array([-5.8824, -2.9412,  0.0000,  2.9412,  5.8824])
    assert_array_almost_equal(result, result_expected, decimal=4)

    result = sim_ann_latency_correction(spike_trains)[0]
    result_expected = np.array([-5.8824, -2.9412,  0.0000,  2.9412,  5.8824])
    assert_array_almost_equal(result, result_expected, decimal=2)

    st1 = spk.SpikeTrain([100, 200, 300], [0, 1000])
    st2 = spk.SpikeTrain([105, 205, 300], [0, 1000])

    STDM1_result = Spike_time_difference_matrix([st1, st2])
    result_expected = [np.array([0.00000000, 3.33333333]), np.array([-3.33333333, 0.00000000])]
    assert_array_almost_equal(STDM1_result, result_expected)

    st3 = spk.SpikeTrain([105, 195, 500], [0, 1000])

    STDM2_result = Spike_time_difference_matrix([st1, st2, st3])
    result_expected = [np.array([0.00000000, 3.33333333, 0.00000000]), np.array([-3.33333333, 0.00000000, -5.00000000]), np.array([-0.00000000, 5.00000000, 0.00000000])]
    assert_array_almost_equal(STDM2_result, result_expected)

    STDM3_result = Spike_time_difference_matrix([st1, st3])
    result_expected = [np.array([0.00000000, 0.00000000]), np.array([-0.00000000, 0.00000000])]
    assert_array_almost_equal(STDM3_result, result_expected)

    result = latency_correction_extrapol(STDM1_result)[0]
    result_expected = np.array([-1.666667, 1.666667])
    assert_array_almost_equal(result, result_expected)

    result = latency_correction_extrapol(STDM2_result)[0]
    result_expected = np.array([-1.111111, 2.777778, -1.666667])
    assert_array_almost_equal(result, result_expected)

    result = latency_correction_extrapol(STDM3_result)[0]
    result_expected = np.array([0, 0])
    assert_array_almost_equal(result, result_expected)

    result = latency_correction_intrapol(STDM1_result)[0]
    result_expected = np.array([-1.666667, 1.666667])
    assert_array_almost_equal(result, result_expected)

    result = latency_correction_intrapol(STDM2_result)[0]
    result_expected = np.array([-1.111111, 2.777778, -1.666667])
    assert_array_almost_equal(result, result_expected)

    result = latency_correction_intrapol(STDM3_result)[0]
    result_expected = np.array([0, 0])
    assert_array_almost_equal(result, result_expected)

    result = cost_matrix([st1, st2, st3])
    result_expected = np.array([[0., 4.082483, 5.], [4.082483, 0., 7.07106781], [5., 7.07106781, 0.]])
    assert_array_almost_equal(result, result_expected)