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
    check_SMR_mesh, \
    SimpleMap2Polar2D, \
    AthenaVTK, \
    split_VTK_and_trim_par, \
    AthenaMultiVTK, \
    AthenaSMRVTK, \
    AthenaLIS, \
    AthenaMultiLIS
from . import constants
from .constants import \
    Constants
