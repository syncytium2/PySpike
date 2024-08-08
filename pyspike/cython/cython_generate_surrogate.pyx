#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

"""
cython_generate_surrogate.pyx

cython implementation of generating surrogate

Copyright 2015, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""

"""
To test whether things can be optimized: remove all yellow stuff
in the html output::

  cython -a cython_generate_surrogate.pyx

which gives::

  cython_generate_surrogate.html

"""

import numpy as np
cimport numpy as np
import random

cdef extern from "numpy/npy_common.h":
    ctypedef int npy_intp

cpdef tuple Spike_Order_surro(np.ndarray[np.int64_t, ndim=2] indies1,
                           np.ndarray[np.int64_t, ndim=1] firsts,
                           np.ndarray[np.int64_t, ndim=1] seconds,
                           int num_swaps):
    
    cdef int num_coins = indies1.shape[1]
    cdef int error_count = 0
    cdef int sc = 0
    cdef int coin, train1, train2, pos1, pos2, i, fc, sedc
    cdef bint brk
    cdef np.ndarray[np.int64_t, ndim=1] fi11, fi21, fi12, fi22, fiu, sed
    cdef np.ndarray[np.int64_t, ndim=1] new_trains

    while sc < num_swaps:
        indies2 = indies1.copy()
        brk = False
        coin = random.randint(0, num_coins-1)
        
        train1 = indies1[1, coin]
        train2 = indies1[2, coin]
        pos1 = indies1[3, coin]
        pos2 = indies1[4, coin]
        
        fi11 = np.where(indies1[3,:] == pos1)[0]
        fi21 = np.where(indies1[4,:] == pos1)[0]
        fi12 = np.where(indies1[3,:] == pos2)[0]
        fi22 = np.where(indies1[4,:] == pos2)[0]
        fiu = np.unique(np.concatenate((fi11, fi21, fi12, fi22)))

        indies1[1, fi11] = train2
        indies1[2, fi21] = train2
        indies1[1, fi12] = train1
        indies1[2, fi22] = train1

        for fc in fiu:
            new_trains = np.sort(indies1[1:3, fc])
            for i in range(len(firsts)):
                if firsts[i] == new_trains[0] and seconds[i] == new_trains[1]:
                    indies1[0, fc] = i
                    break
        for fc in fiu:
            sed = np.setdiff1d(np.where(indies1[0, :] == indies1[0, fc])[0], fc)
            for sedc in range(len(sed)):
                if len(np.intersect1d(indies1[3:5, sed[sedc]], indies1[3:5, fc])) > 0:
                    error_count += 1
                    indies1 = indies2
                    brk = True
                    break
            if brk:
                break
        if brk:
            if error_count <= num_coins:
                continue
            else:
                sc = num_swaps
        sc += 1
    return indies1, error_count