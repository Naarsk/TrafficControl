"""Build script for the Cython extension `src._build_kernel`.

Usage (one-time, from the project root, after `pip install -r requirements.txt`):

    python setup.py build_ext --inplace

This compiles `src/_build_kernel.pyx` into a platform-specific shared
library (`src/_build_kernel*.pyd` on Windows, `*.so` on Linux/macOS).
The Python code in `src/model.py` imports the compiled extension if
present and falls back to the pure-Python build otherwise.
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension(
        name="src._build_kernel",
        sources=["src/_build_kernel.pyx"],
        include_dirs=[np.get_include()],
    ),
]


setup(
    name="trafficcontrol-mdp",
    version="0.1.0",
    ext_modules=cythonize(extensions, language_level="3"),
    zip_safe=False,
)
