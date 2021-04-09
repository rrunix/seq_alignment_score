from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

extra_compile_args = ['-O3', '-ffast-math', '-march=native', '-fopenmp']
extra_link_args=['-fopenmp']


exts = [
    Extension('sequence_metrics.needleman_wunsch', ['sequence_metrics/needleman_wunsch.pyx'],
              extra_compile_args=extra_compile_args, extra_link_args=extra_link_args),
]

compiler_directives = {
    'boundscheck': False,
    'wraparound': False,
    'nonecheck': False,
    'cdivision': True,
    'profile': True
}

setup(name='sequence_metrics',
      version="0.1alpha",
      ext_modules= cythonize(exts, compiler_directives=compiler_directives),
      include_dirs=[np.get_include()],
      packages=['sequence_metrics'],
      install_requires=[
          'numpy>=1.11.0', 'Cython>=0.29.14'
      ],
      author="rrunix",
      author_email="ruben.rrf93@gmail.com",
      license="BSD",
      description="A bunch of metrics between two sequences",
      keywords="Sequence metrics")
