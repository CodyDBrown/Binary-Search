from distutils.core import setup
from Cython.Build import cythonize
import Cython

print(Cython.__version__)
setup( ext_modules = cythonize("CyBinaryParameters.pyx"))