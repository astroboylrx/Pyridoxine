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
    float32_to_float24, \
    float24_to_float32, \
    pack_float24_to_uint8, \
    unpack_uint8_to_float24, \
    convert_array_float32_to_float24, \
    convert_array_float24_to_float32, \
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
    trim_VTK, \
    AthenaMultiVTK, \
    AthenaSMRVTK, \
    AthenaLIS, \
    trim_LIS, \
    AthenaMultiLIS
from . import constants
from .constants import \
    Constants
