from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("cythonize\det_n2.pyx")
)