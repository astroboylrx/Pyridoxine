""" utility: snippets served as utilities

    This is a subpackage of pyridoxine.
    This package, "utility", includes various scripts to serve other dedicated subpackages
"""

from . import vec
from .vec import \
    Vector
from . import rw
from .rw import \
    loadtxt, \
    readbin, \
    loadbin, \
    AthenaVTK
