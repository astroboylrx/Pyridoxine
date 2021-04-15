""" utility: snippets served as utilities

    This is a subpackage of pyridoxine.
    This package, "utility", includes various scripts to serve other dedicated subpackages
"""

from . import vec
from .vec import \
    Vector, \
    minmax, \
    d, \
    fourier_amp
from . import rw
from .rw import \
    loadtxt, \
    readcol, \
    readbin, \
    writebin, \
    loadbin, \
    dumpbin, \
    SimpleMap2Polar2D, \
    AthenaVTK, \
    AthenaMultiVTK, \
    AthenaSMRVTK, \
    AthenaLIS, \
    AthenaMultiLIS
from . import constants
from .constants import \
    Constants
