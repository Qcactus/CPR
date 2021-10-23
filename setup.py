import os
import numpy as np
from setuptools import setup, Extension
from Cython.Build import cythonize

setup(
    ext_modules=cythonize(
        Extension(
            "*",
            ["recq/cyutils/*.pyx"],
            language="c++",
            extra_compile_args=["-std=c++11"],
        ),
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
        },
    ),
    include_dirs=[np.get_include()],
)
