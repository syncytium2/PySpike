"""
Copyright 2014-2018, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License
"""

from __future__ import absolute_import

__all__ = ["isi_distance", "spike_distance", "spike_sync", "psth",
           "spikes", "spike_directionality", "SpikeTrain",
           "PieceWiseConstFunc", "PieceWiseLinFunc", "DiscreteFunc"]

from .PieceWiseConstFunc import PieceWiseConstFunc
from .PieceWiseLinFunc import PieceWiseLinFunc
from .DiscreteFunc import DiscreteFunc
from .SpikeTrain import SpikeTrain

from .isi_distance import isi_profile, isi_distance, isi_profile_multi,\
    isi_distance_multi, isi_distance_matrix
from .spike_distance import spike_profile, spike_distance, spike_profile_multi,\
    spike_distance_multi, spike_distance_matrix
from .spike_sync import spike_sync_profile, spike_sync,\
    spike_sync_profile_multi, spike_sync_multi, spike_sync_matrix,\
    filter_by_spike_sync
from .psth import psth

from .spikes import load_spike_trains_from_txt, save_spike_trains_to_txt, \
    spike_train_from_string, import_spike_trains_from_time_series, \
    merge_spike_trains, generate_poisson_spikes

from .spike_directionality import spike_directionality, \
    spike_directionality_values, spike_directionality_matrix, \
    spike_train_order_profile, spike_train_order_profile_bi, \
    spike_train_order_profile_multi, spike_train_order, \
    spike_train_order_bi, spike_train_order_multi, \
    optimal_spike_train_sorting, permutate_matrix

# Expose the installed version via importlib.metadata (stdlib since 3.8).
# Previously this used pkg_resources, which depends on setuptools — and from
# Python 3.12 onwards venvs no longer ship setuptools by default, so the old
# import would fail at runtime in fresh 3.12+ environments.
from importlib.metadata import version as _get_version, PackageNotFoundError

try:
    __version__ = _get_version("pyspike")
except PackageNotFoundError:
    # Running from a source checkout that hasn't been installed.
    __version__ = "0.0.0+unknown"

disable_backend_warning = False

def NoCythonWarn():
    """ Warn exactly once
         (called when an import of one of the cython_...so modules failed)
    """
    global disable_backend_warning  # initialized False in __init__.py
    if not disable_backend_warning:
        print("Warning: Cython implementation not found." +
            " Make sure that PySpike is installed by running\n" +
            " 'python setup.py build_ext --inplace'\n" +
            "Falling back to slow python backend.\n")
    disable_backend_warning = True
