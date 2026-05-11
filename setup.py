""" setup.py

Compile the Cython extensions for PySpike.

All packaging metadata (name, version, dependencies, classifiers, ...) lives in
pyproject.toml. This file only declares the C extension modules, because that
still needs imperative setup() configuration.

To compile cython files in-place:
    python setup.py build_ext --inplace


Copyright 2014-2026, Mario Mulansky <mario.mulansky@gmx.net>

Distributed under the BSD License

"""
import os.path

from setuptools import Extension, setup

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True


class numpy_include(os.PathLike):
    """Defers import of numpy until the build environment is in place.

    pyproject.toml lists numpy as a build-system requirement, so by the time
    setup.py actually runs build_ext, numpy is importable. We can't import it
    at module top level, though, because setuptools imports setup.py before
    build-system requires are installed.
    """

    def __str__(self):
        import numpy
        return numpy.get_include()

    def __fspath__(self):
        return str(self)


_CYTHON_MODULES = (
    "cython_add",
    "cython_get_tau",
    "cython_profiles",
    "cython_distances",
    "cython_directionality",
    "cython_simulated_annealing",
)


def _all_c_sources_present():
    return all(
        os.path.isfile(f"pyspike/cython/{name}.c") for name in _CYTHON_MODULES
    )


use_c = _all_c_sources_present()

if not use_cython and not use_c:
    print("Cython not installed and no pre-generated .c files found. "
          "PySpike will fall back to the pure-Python backend (slow).")

cmdclass = {}
ext_modules = []

if use_cython:  # Cython is available, compile .pyx -> .c -> binary
    ext_modules = [
        Extension(f"pyspike.cython.{name}", [f"pyspike/cython/{name}.pyx"])
        for name in _CYTHON_MODULES
    ]
    cmdclass["build_ext"] = build_ext
elif use_c:  # No Cython, but pre-generated .c files are present
    ext_modules = [
        Extension(f"pyspike.cython.{name}", [f"pyspike/cython/{name}.c"])
        for name in _CYTHON_MODULES
    ]
# else: neither Cython nor .c files — fall through to pure-Python backend.

setup(
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    include_dirs=[numpy_include()],
)
