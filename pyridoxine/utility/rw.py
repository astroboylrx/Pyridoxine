""" Provide convenient reading/writing functions """

import traceback
import struct
from array import array
from numbers import Number
import subprocess as subp
import numpy as np
import scipy.interpolate as spint
import numba
from numba import prange
import ast
import pandas as pd
import shapely.geometry as g
import os
import copy
import warnings
from io import StringIO
import matplotlib.pyplot as plt
from ..plt import plt_params, ax_labeling

__valid_array_typecode = ['b', 'B', 'u', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'f', 'd']


def __read_formatted_column_super_slow(filepath, col2read):
    """ Read one data column from a formatted text file.
        This is still way more slower than np.loadtxt().
        Instead of building wheels, I found np.genfromtxt() is even faster.
        Then I found pandas.read_csv(...).as_matrix() is way more faster.
        See also: loadtxt() below and readcol() below
    """

    line_length = 0
    num_lines = int(subp.getoutput("wc -l " + filepath).split()[0])
    num_header_lines = 0

    with open(filepath) as ascii_data:
        for line in ascii_data:
            if line[0] == '#':
                num_lines -= 1
                num_header_lines += 1
                continue
            else:
                line_length = len(line)
                a = line.split()
                if col2read >= len(a):
                    raise ValueError("There is no such column to read: ", col2read)
                break

    if line_length == 0:
        raise ValueError("It seems there is nothing to read, line_length = ", line_length)

    ascii_data = open(filepath)
    for i in range(num_header_lines):
        ascii_data.readline()

    first_data_line = ascii_data.readline()
    goods = 0
    idx = 0
    col_start = 0
    col_end = 0

    while idx < len(first_data_line):
        if first_data_line[idx] != ' ':
            while idx < len(first_data_line) and first_data_line[idx] != ' ':
                idx += 1
            goods += 1
            continue
        else:
            if goods == col2read:
                col_start = idx
            if goods == col2read+1:
                col_end = idx
                break
            while idx < len(first_data_line) and first_data_line[idx] == ' ':
                idx += 1
        continue
    if goods == col2read+1: # in case at the end of line
        col_end = idx

    col_width = col_end - col_start
    data_location = ascii_data.tell() - line_length + col_start
    ascii_data.seek(data_location, 0)

    c = np.zeros(num_lines)
    for i in range(num_lines):
        c[i] = float(ascii_data.read(col_width))
        data_location += line_length
        ascii_data.seek(data_location, 0)
        # seeking from current location yield "UnsupportedOperation: can't do nonzero cur-relative seeks"

    """ Here is an example of time-consuming part. 
    Timer unit: 1e-06 s
    Total time: 0.674234 s    
     Hits         Time  Per Hit   % Time  Line Contents
     ==============================================================
     ...
        1      12504.0  12504.0      1.9      num_lines = int(sp.getoutput("wc -l "+filepath).split()[0])
     ...
    35261      25232.0      0.7      3.7      for i in range(num_lines):
    35260     384296.0     10.9     57.0          c[i] = float(ascii_data.read(col_width))
    35260      26407.0      0.7      3.9          data_location += line_length
    35260     224863.0      6.4     33.4          ascii_data.seek(data_location, 0)
    
    As a comparison, to read the entire file, np.loadtxt() used 402 ms, np.genfromtxt used 577 ms,
    while pd.read_csv(..., delim_whitespace=True, header=None).as_matrix() used 141 ms.
    """
    return c


@numba.njit
def float32_to_float24(float32_val):

    # Method 1 to find the closest representation of float24 in float32 plus handle mantissa overflow
    # Create a very small float32 number with rounding constant (1<<5) in the mantissa (=32*2**-23)
    # View Binary representation for (32 * 2 ** -23) as a float32
    # Add the rounding constant to the original float32 value
    # float32_val += np.uint32(0x36800000).view(np.float32)
    # Okay, this does not work b/c mantissa rounding assumes same sign bit, same exponent bits,
    # but 1<<32 means different numbers when sign/exponent changes, so there is no such a magic number

    # Convert float32 to IEEE 754 bits (32-bit unsigned integer)
    f32_bits = np.float32(float32_val).view(np.uint32)

    # Special case: handle zero explicitly
    if float32_val == 0.0:  # this includes both 0.0 and -0.0
        return np.uint32((f32_bits >> 31) << 23)  # Preserve the sign bit in float24

    #if float32_val == 0x80000000:  # -0.0 in float32 bit representation
    #    return np.uint32(0x800000)  # Equivalent to 1 << 23

    # Extract sign, exponent, and mantissa from float32
    sign = (f32_bits >> 31) & 0x1
    exponent = (f32_bits >> 23) & 0xFF
    mantissa = f32_bits & 0x7FFFFF

    # Handle special cases for float32: NaN and Inf
    if exponent == 0xFF:  # Exponent is all 1s in float32
        if mantissa == 0:
            # Float32 Inf, represent it as float24 Inf (all 1s in exponent, 0 mantissa)
            return (sign << 23) | (0x3F << 17)  # Exponent all 1s (6-bit), mantissa 0
        else:
            # Float32 NaN, represent it as float24 NaN (all 1s in exponent, non-zero mantissa)
            # Ensure the mantissa is non-zero in float24, explicitly set the least significant bit
            return (sign << 23) | (0x3F << 17) | ((mantissa >> 6) | 0x1)  # Exponent all 1s, set least bit of mantissa

    # Method 2 to find the closest represenation of float24 in float32 plus handle mantissa overflow
    # Rounding: Add a rounding constant to the mantissa before splitting exponent/mantissa
    mantissa += np.uint32(0x00000020)  # =1<<5, half of the bits we are truncating (2^5 = 32)
    if mantissa >= np.uint32(0x00800000):  # =1<<23, if true, then mantissa overflow, should increase the exponent
        mantissa = 0  # Reset mantissa after rounding overflow
        exponent += 1  # Increase exponent to account for the overflow

    # Convert to float24 (6-bit exponent, 17-bit mantissa)
    float24_exponent = exponent - 127 + 31  # Adjust the exponent bias from 127 (float32) to 31 (float24)

    # Handle underflow (too small exponent results in zero)
    if float24_exponent <= 0:
        return np.uint32(sign << 23)  # Return zero with the sign bit

    # Handle overflow, cap exponent if it exceeds 6 bits
    if float24_exponent >= 63:
        # Set exponent to all 1s (Inf representation)
        return (sign << 23) | (0x3F << 17)

    # Now shift mantissa to fit into 17 bits
    mantissa = mantissa >> 6

    # Pack the sign, exponent, and mantissa into 24 bits (shift mantissa by 6 to fit 17 bits)
    float24_bits = (sign << 23) | (float24_exponent << 17) | mantissa

    return float24_bits


@numba.njit
def float24_to_float32(float24_val):
    # Special case: handle zero explicitly
    if float24_val == 0:
        return np.float32(0.0)
    if float24_val == 0x800000:  # =1<<23, which is -0.0 in float24 representation
        return np.float32(-0.0)

    # Extract sign, exponent, and mantissa from float24
    sign = (float24_val >> 23) & 0x1
    exponent = (float24_val >> 17) & 0x3F  # 6-bit exponent
    mantissa = float24_val & 0x1FFFF  # 17-bit mantissa

    # Handle special cases for float24: Inf and NaN
    if exponent == 0x3F:  # Exponent all 1s in float24 (special case: Inf or NaN)
        if mantissa == 0:
            # Inf case, return float32 Inf
            return np.uint32((sign << 31) | (0xFF << 23)).view(np.float32)
        else:
            # NaN case, return float32 NaN
            # Ensure mantissa is non-zero after shifting (set least significant bit)
            return np.uint32((sign << 31) | (0xFF << 23) | ((mantissa << 6) | 0x1)).view(np.float32)

    # Normal case: convert back to float32 (6-bit exponent, 23-bit mantissa)
    float32_exponent = exponent - 31 + 127  # Adjust the exponent bias from 31 (float24) to 127 (float32)

    # Reconstruct the float32 by shifting the mantissa back to 23 bits
    float32_bits = (sign << 31) | (float32_exponent << 23) | (mantissa << 6)

    return np.uint32(float32_bits).view(np.float32)


@numba.njit
def pack_float24_to_uint8(float24_val):
    """ Split 24-bit float into three 8-bit integers (stored in 3 uint8 values) """

    byte1 = (float24_val >> 16) & 0xFF  # Highest 8 bits
    byte2 = (float24_val >> 8) & 0xFF  # Middle 8 bits
    byte3 = float24_val & 0xFF  # Lowest 8 bits
    return byte1, byte2, byte3


@numba.njit
def unpack_uint8_to_float24(byte1, byte2, byte3):
    """ Combine three 8-bit integers into a 24-bit float """

    float24_val = (byte1 << 16) | (byte2 << 8) | byte3  # this auto convert them to np.int64
    return float24_val


@numba.njit
def convert_array_float32_to_float24(arr):
    """ Use a 2D uint8 array to store the 24-bit data: each row has 3 uint8 values (3 bytes for each float24) """

    result = np.zeros((arr.size, 3), dtype=np.uint8)
    for i in prange(arr.size):
        result[i] = pack_float24_to_uint8(float32_to_float24(arr[i]))
    return result


@numba.njit
def convert_array_float24_to_float32(arr):
    """ Convert a 2D uint8 array that stores float24 data to a 1D float32 array """

    result = np.zeros(arr.shape[0], dtype=np.float32)
    for i in prange(arr.shape[0]):
        result[i] = float24_to_float32(unpack_uint8_to_float24(arr[i, 0], arr[i, 1], arr[i, 2]))
    return result


def get_minimum_unsigned_dtype(max_value):
    """ Get the minimum unsigned dtype that can hold the max_value. """

    if max_value <= np.iinfo(np.uint8).max:
        return 'u1'  # uint8
    elif max_value <= np.iinfo(np.uint16).max:
        return 'u2'  # uint16
    elif max_value <= np.iinfo(np.uint32).max:
        return 'u4'  # uint32
    else:
        return 'u8'  # uint64


def get_minimum_signed_dtype(min_value, max_value):
    """ Get the minimum signed dtype that can hold the min_value and max_value. """

    if min_value >= 0:
        # If the minimum value is non-negative, use unsigned integers
        return get_minimum_unsigned_dtype(max_value)

    if min_value >= np.iinfo(np.int8).min and max_value <= np.iinfo(np.int8).max:
        return 'i1'  # int8
    elif min_value >= np.iinfo(np.int16).min and max_value <= np.iinfo(np.int16).max:
        return 'i2'  # int16
    elif min_value >= np.iinfo(np.int32).min and max_value <= np.iinfo(np.int32).max:
        return 'i4'  # int32
    else:
        return 'i8'  # int64


def write_dtype_to_file(dtype, f):
    """ Writes the numpy dtype to a binary file as encoded text """

    num_fields = len(dtype.descr)
    f.write(f"num_fields: {num_fields}\n".encode('utf-8'))

    # Write each field in the dtype
    for field in dtype.descr:
        field_name, field_type = field[:2]
        if len(field) == 3:  # If there's a shape (like ('pos', 'u1', 9))
            shape = field[2]
            field_str = f"[{field_name}, {field_type}, {shape}]\n"
        else:  # If there's no shape (like ('property_index', 'i4'))
            field_str = f"[{field_name}, {field_type}]\n"
        f.write(field_str.encode('utf-8'))


def read_dtype_from_file(f):
    """Reads a numpy dtype from encoded text in a binary file"""

    fields = []
    # Read the first line to get the number of fields
    first_line = f.readline().decode('utf-8').strip()
    if not first_line.startswith("num_fields:"):
        raise ValueError("The file does not contain the expected number of fields header.")
    num_fields = int(first_line.split(":")[1].strip())

    # Read each subsequent line to get the dtype fields
    for _ in range(num_fields):
        line = f.readline().decode('utf-8').strip()
        # Remove square brackets and then Split the line by commas
        decoded_line = line.strip("[]")
        first_comma_idx = decoded_line.find(',')
        field_name = decoded_line[:first_comma_idx].strip()
        remainder = decoded_line[first_comma_idx + 1:].strip()
        second_comma_idx = remainder.find(',')
        if second_comma_idx == -1:  # No shape, just field_type
            field_type = remainder.strip()
            fields.append((field_name, field_type))
        else:
            field_type = remainder[:second_comma_idx].strip()
            shape_str = remainder[second_comma_idx + 1:].strip()
            # Convert the shape string back to a tuple (e.g., '(9,)' becomes (9,))
            shape = ast.literal_eval(shape_str)
            fields.append((field_name, field_type, shape))
    return np.dtype(fields)


def loadtxt(filepath, h=0, c=None):
    """ Wrapping pandas.read_csv()
        h: how many rows to skip in order to reach real data
        c: which column(s) to read
    """

    if c is None:
        return pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=h).values
    else:
        if isinstance(c, Number):
            c = [c]
        return pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=h, usecols=c).values


def readcol(filename, col2read):
    """ Read data by columns from a formmated text file
    :param filename: the file name
    :param col2read: the column(s) to read, e.g., 3, or [2,3]
    :return: data in numpy ndarray
    """

    if isinstance(col2read, Number):
        col2read = '$'+str(col2read)
    if isinstance(col2read, (list, tuple, np.ndarray)):
        col2read = ",".join(['$'+str(x) for x in np.asarray(col2read).flatten()])

    try:
        data_str = subp.check_output(["bash", "-c", "awk '{print "+col2read+"}' "+filename], stderr=subp.STDOUT).decode('utf-8')
    except Exception:
        traceback.print_exc()
        return None

    return loadtxt(StringIO(data_str))
    #return np.loadtxt(StringIO(data_str))


def readbin(file_handler, dtype='d'):
    """ Retrieve a certain length of data from a binary file
    :param file_handler: an opened file object
    :param dtype: data type, format string
    :return: re-interpreted data
    """

    if not isinstance(dtype, str):
        raise TypeError("reading format need to be a str: ", dtype)

    ori_pos = file_handler.tell()
    try:
        data = struct.unpack(dtype, file_handler.read(struct.calcsize(dtype)))
        if len(data) == 1:
            return data[0]
        return data
    except Exception:
        traceback.print_exc()
        print("Rolling back to the original stream position...")
        file_handler.seek(ori_pos)
        return None


def writebin(file_handler, data, dtype='d'):
    """ Write a certain length of data to a binary file
    :param file_handler: an opened file object
    :param data: stuff to write
    :param dtype: data type, format string
    """

    if not isinstance(dtype, str):
        raise TypeError("writing format need to be a str: ", dtype)

    ori_pos = file_handler.tell()
    try:
        file_handler.write(struct.pack(dtype, data))
        return True
    except Exception:
        traceback.print_exc()
        print("Rolling back to the original stream position...")
        file_handler.seek(ori_pos)
        return False


def loadbin(file_handler, dtype='d', num=1):
    """ Load a sequence of data from a binary file, return a ndarray """

    if dtype not in __valid_array_typecode:
        raise ValueError("bad typecode: "+dtype+" (must be one of ["+",".join(__valid_array_typecode)+"])")
    if not isinstance(num, Number):
        raise TypeError("size should be a scalar, but got: ", num)
    if num < 0:
        raise ValueError("size should be larger than 0, but got: "+str(num))

    data = array(dtype)
    ori_pos = file_handler.tell()
    try:
        data.fromfile(file_handler, num)
        data = np.asarray(data)
        return data
    except Exception:
        traceback.print_exc()
        print("Rolling back to the original stream position...")
        file_handler.seek(ori_pos)
        return None


def dumpbin(file_handler, data, dtype='d'):
    """ Dump a sequence of data to a binary file """

    pass


def check_SMR_mesh(Nx, Lx=1, nlev2check=6,
                   Nx2check = [64, 96, 128, 192, 256, 384, 512, 768, 1024]):
    """ check if the desired mesh works with SMR """

    Nx = np.atleast_1d(np.asarray(Nx))
    if Nx.size > 1:
        dx = Lx / Nx[0]
        for i in range(0, Nx.size):
            disp = (Nx[0] * 2 ** i - Nx[i]) / 2
            workable = (int(disp / 2 ** i) == (disp / 2 ** i))
            print("Level "+str(i)+": Nx = "+"{:4d}".format(Nx[i])+"("+"{:6.1f}".format(Nx[i]/2**i)+" x2^i)"
                  +", lx="+"{:8.4f}".format(dx/2**i * Nx[i])+", dx="+"{:.6e}".format(dx/2**i)
                  +", disp = "+"{:9.1f}".format(disp)
                  +", disp/2^"+str(i)+" = "+"{:6.2f}".format(disp / 2 ** i)+", usable: "+str(workable))
        print("Total # of cells: "+str(np.sum(Nx ** 2)))
    elif Nx.size == 1:
        dx = Lx / Nx[0]
        for i, lev in enumerate(list(range(1, nlev2check))):
            finer_Nx = Nx[0] * 2**lev
            tmp_Nx2check = [x for x in Nx2check if x < finer_Nx]
            disp = [(finer_Nx - x) / 2 for x in Nx2check if x < finer_Nx]
            if len(disp) > 0:
                selection = [True if (x.is_integer() and (x/2**lev).is_integer()) else False for x in disp]
                tmp_Nx2check = [x for x, y in zip(tmp_Nx2check, selection) if y is True]
                disp = [int(x) for x, y in zip(disp, selection) if y is True]
                if len(disp) > 0:
                    lx = [x * dx/2**lev for x in tmp_Nx2check]
                    lx = [round(x, 3 - int(np.floor(np.log10(abs(x)))) - 1) for x in lx]
                    print("Level " + str(lev) + ": dx=" + "{:.6e}".format(dx / 2 ** lev) + "; Available Nx and disp: ", list(zip(tmp_Nx2check, disp, lx)))
                else:
                    print("Level " + str(lev) + ": dx=" + "{:.6e}".format(dx / 2 ** lev) + "; NO available Nx")


class SimpleMap2Polar2D:

    def __init__(self, x, y, data, r, t,
                 origin=np.array([0, 0]), orders=(1, 1), data_names=None):
        """ Map the grid data to polar data
            :param x, y: 1D array, the Cartesian coordinates
            :param data: 2D array or list of 2D array, original Cartesian data
            :param r, t: 1D array, the Polar coordinates
            :param origin: origin of the Polar coordinates
            :param orders: interpolation order in x and y direction
            :param data_names: if not None, polar_data is returned as dictionary
        """

        self.r, self.t = r, t
        self.dr, self.dt = np.diff(r).mean(), np.diff(t).mean()
        self.r_max = self.r[-1] + np.diff(r).mean() / 2.0
        self.origin = np.asarray(origin)
        if self.origin.size != 2:
            raise ValueError("origin must be a 2-element array/list/tuple")
        self.names = data_names
        if self.names is not None:
            if isinstance(self.names, str):
                self.names = [self.names]
            if isinstance(data, (tuple, list)):
                if len(self.names) < len(data):
                    raise ValueError("length mismatch between data and data_names")
                elif len(self.names) > len(data):
                    print("Warning: data_names more than len(data):", data_names, len(data))
                else:
                    pass

        # upper case variables to denotes mesh
        R, T = np.meshgrid(r, t)
        self.new_X = R * np.cos(T) + origin[0]
        self.new_Y = R * np.sin(T) + origin[1]

        if self.names is None and isinstance(data, np.ndarray):
            # N.B., non-linear interpolation method would result in negative values near the shock edges!!!!
            interp_spline = spint.RectBivariateSpline(y, x, data, kx=orders[0], ky=orders[1])
            # N.B., using Y, X gives quantities in the right direction without transpose
            # REFS: https://scipython.com/book/chapter-8-scipy/examples/two-dimensional-interpolation-with-scipyinterpolaterectbivariatespline/
            self.polar_data = interp_spline.ev(self.new_Y, self.new_X)

            # in this case, b/c we are interpolating to non-rect grid, the speed doesn't change too much
            # self.polar_data = interp_spline(self.new_Y, self.new_X, grid=False)

        elif self.names is not None and isinstance(data, np.ndarray):
            interp_spline = spint.RectBivariateSpline(y, x, data, kx=orders[0], ky=orders[1])
            self.polar_data = dict()
            self.polar_data[self.names[0]] = interp_spline.ev(self.new_Y, self.new_X)

        elif isinstance(data, (tuple, list)):
            if self.names is None:
                self.polar_data = []
            else:
                self.polar_data = dict()

            for idx, item in enumerate(data):
                interp_spline = spint.RectBivariateSpline(y, x, item, kx=orders[0], ky=orders[1])
                if self.names is None:
                    self.polar_data.append(interp_spline.ev(self.new_Y, self.new_X))
                else:
                    self.polar_data[self.names[idx]] = interp_spline.ev(self.new_Y, self.new_X)
        else:
            raise NotImplementedError("Unknown type for data" + str(type(data)))

    def __getitem__(self, data_name):
        """ Overload indexing operator [] """

        if not isinstance(self.polar_data, dict):
            raise TypeError("No data_names assigned. Dictionary-access is not supported.")

        if data_name in self.polar_data:
            return self.polar_data[data_name].view()
        else:
            raise KeyError(data_name+" not found. Available are "+", ".join(self.names))


class AthenaVTK:
    """ Read data from VTK files from SI simulations (by Athena)
        AthenaVTK is able to read BINARY data of STRUCTURED_POINTS, either 2D or 3D
        e.g.,
            >>> a = AthenaVTK("Cout.0500.vtk", silent=False)
            Read [density, momentum, particle_density, particle_momentum] at Nx=[64, 64, 64]
    """

    def __init__(self, filename, wanted=None, xyz_order=None, silent=True):
        """
        directly read binary data from file
        :param filename: the single VTK file
        :param wanted: specify the desired data names (to avoid loading all of them)
        :param xyz_order: useful when data(vector) is organized as (x, z, y)
        :param silent: whether print out basic info or not
        """

        if xyz_order is None:
            self.__xyz_order = {'x': 0, 'y': 1, 'z': 2}
        elif xyz_order == "xz":
            self.__xyz_order = {'x': 0, 'y': 2, 'z': 1}
        else:
            self.__xyz_order = xyz_order

        if isinstance(wanted, str):  # None is not a str
            wanted = [wanted]
        if isinstance(wanted, list):
            if len(wanted) == 0:
                wanted = None

        f = open(filename, 'rb')
        eof = f.seek(0, 2)  # record eof position
        f.seek(0, 0)

        tmp_line = f.readline()  # normally "vtk DataFile Version 3.0"
        tmp_line = f.readline().decode('utf-8')  # Really cool Athena data at time= 0.000000e+00, level= 0, domain= 0
        self.t = float(tmp_line[tmp_line.find("time=")+6:tmp_line.find(", level=")])
        self.level = int(tmp_line[tmp_line.find("level=")+7:tmp_line.find(", domain=")])
        self.domain = int(tmp_line[tmp_line.find("domain=")+8:])

        tmp_line = f.readline().decode('utf-8')[:-1]  # get rid of '\n'
        if tmp_line != "BINARY":
            raise TypeError("This VTK file does not contain binary data, we got ", tmp_line)
        tmp_line = f.readline().decode('utf-8')[:-1]  # get rid of '\n'
        if tmp_line != "DATASET STRUCTURED_POINTS":
            raise TypeError("This VTK file has a dataset of '"+tmp_line+"', which we cannot handle")

        tmp_line = f.readline().decode('utf-8')  # for example, "DIMENSIONS 129 65 1"
        self.Nx = np.array([int(x) - 1 for x in tmp_line[11:].split()])
        if self.Nx[2] == 0:
            self.dim = 2
        else:
            self.dim = 3

        tmp_line = f.readline().decode('utf-8')
        assert (tmp_line[:6] == "ORIGIN"), "no ORIGIN info: "+tmp_line
        self.left_corner = np.array([float(x) for x in tmp_line[7:].split()])

        tmp_line = f.readline().decode('utf-8')
        assert (tmp_line[:7] == "SPACING"), "no SPACING info: "+tmp_line
        self.dx = np.array([float(x) for x in tmp_line[8:].split()])

        self.ccx = np.zeros(self.Nx[0])
        self.ccy = np.zeros(self.Nx[1])
        if self.dim == 3:
            self.ccz = np.zeros(self.Nx[2])
        self._set_cell_centers_()
        self.right_corner = self.left_corner + self.dx * self.Nx
        self.box_size = self.dx * self.Nx
        self.box_min = copy.deepcopy(self.left_corner)
        self.box_max = copy.deepcopy(self.right_corner)
        if self.dim != np.count_nonzero(self.box_max - self.box_min):
            raise RuntimeError("the dimension of the data is confusing, box_min=",
                               self.box_min, ", box_max=", self.box_max, ", dim=", self.dim)

        tmp_line = f.readline().decode('utf-8')
        assert(tmp_line[:9] == "CELL_DATA"), "no CELL_DATA info: "+tmp_line
        self.size = int(tmp_line[10:])
        self.num_cells = self.size

        self.svtypes = []
        self.names = []
        self.dtypes = []
        self.data = dict()

        # define ghost zone related variables
        self.ghost_width = 0
        self.data_gh = dict()  # data with ghost zone
        self.ccx_gh = copy.deepcopy(self.ccx)
        self.ccy_gh = copy.deepcopy(self.ccy)
        if self.dim == 3:
            self.ccz_gh = copy.deepcopy(self.ccz)
        self.left_corner_gh = copy.deepcopy(self.left_corner)
        self.right_corner_gh = copy.deepcopy(self.right_corner)

        # define common data name in case wanted uses them
        self.__simplified_names = {
            "dpar": ["particle_density", "rhop", "dpar", "rhopz", "rhopy"],
            #  "particle_density": ["dpar"], comment out to not confuse short/long names
            "rhop": ["particle_density", "dpar", "rhopz", "rhopy"],
            "rhog": ["density"],
            "u": ["momentum", "velocity"],
            "v": ["particle_momentum"],
            "w": ["particle_momentum"],
            "E": ["total_energy"],
            "P": ["pressure"],
            "B": ["cell_centered_B"],
            "pot": ["gravitational_potential"],
            "ppot": ["particle_selfg_potential"]
        }
        self._simplified_names = self.__simplified_names
        self.__simplified_components = ["ux", "uy", "uz",
                                        "vx", "vy", "vz",
                                        "wx", "wy", "wz",
                                        "Bx", "By", "Bz"]
        self.__common_types = {'float': 'f', 'double': 'd', 'int': 'i', 'unsigned_char': 'B'}
        real_wanted = []
        if wanted is not None:
            for i, item in enumerate(wanted):
                if item in self.__simplified_names:
                    real_wanted += self.__simplified_names[item]
                elif item in self.__simplified_components:
                    real_wanted += self.__simplified_names[item[0]]
                else:
                    real_wanted += [item]

        while f.tell() != eof:
            tmp_line = f.readline().decode('utf-8')
            if tmp_line == '\n':
                tmp_line = f.readline().decode('utf-8')
            tmp_line = tmp_line.split()
            fp24_flag = False
            if tmp_line[1][-5:] == "_fp24" and tmp_line[2] == "unsigned_char":
                tmp_line[1] = tmp_line[1][:-5]
                fp24_flag = True
            try:
                type_char = self.__common_types[tmp_line[2]]
            except KeyError:
                warnings.warn("Unrecognized type: " + tmp_line[2] + " for data named "
                              + tmp_line[1] + ". Use float for now.")
                type_char = 'f'
            if wanted is not None:
                if tmp_line[1] not in real_wanted:
                    if tmp_line[0] == "SCALARS":
                        f.readline()  # skip "LOOKUP_TABLE default"
                        f.seek(self.size * struct.calcsize(type_char), 1)
                    if tmp_line[0] == "VECTORS":
                        f.seek(3 * self.size * struct.calcsize(type_char), 1)
                    continue

            self.svtypes.append(tmp_line[0])
            self.names.append(tmp_line[1])
            if fp24_flag:
                tmp_line[2] = "float"
            self.dtypes.append(tmp_line[2])
            if tmp_line[0] == "SCALARS":
                f.readline()  # skip "LOOKUP_TABLE default"
                if fp24_flag:
                    tmp_data = array(type_char)
                    tmp_data.fromfile(f, self.size * 3)
                    tmp_data = convert_array_float24_to_float32(np.asarray(tmp_data).byteswap().reshape([-1, 3]))
                    self.data[tmp_line[1]] = tmp_data.reshape(np.flipud(self.Nx[:self.dim]))
                else:
                    tmp_data = array(type_char)
                    tmp_data.fromfile(f, self.size)
                    self.data[tmp_line[1]] = np.asarray(tmp_data).byteswap().reshape(np.flipud(self.Nx[:self.dim]))

            elif tmp_line[0] == "VECTORS":
                if fp24_flag:
                    tmp_data = array(type_char)
                    tmp_data.fromfile(f, self.size * 3 * 3)  # even for 2D simulations, the vector fields are 3D
                    tmp_data = convert_array_float24_to_float32(np.asarray(tmp_data).byteswap().reshape([-1, 3]))
                    tmp_shape = np.hstack([np.flipud(self.Nx[:self.dim]), 3])
                    self.data[tmp_line[1]] = tmp_data.reshape(tmp_shape)
                else:
                    tmp_data = array(type_char)
                    tmp_data.fromfile(f, self.size*3)  # even for 2D simulations, the vector fields are 3D
                    tmp_shape = np.hstack([np.flipud(self.Nx[:self.dim]), 3])
                    self.data[tmp_line[1]] = np.asarray(tmp_data).byteswap().reshape(tmp_shape)

            else:
                raise NotImplementedError("Dataset attribute "+tmp_line[0]+"not supported.")

        f.close()
        if not silent:
            print("Read ["+", ".join(self.names)+"] at Nx=["+", ".join([str(x) for x in self.Nx])+"]")

    def _set_cell_centers_(self):
        """ Calculate the cell center coordinates """

        self.ccx = np.linspace(self.left_corner[0] + self.dx[0] * (1 - 0.5),
                               self.left_corner[0] + self.dx[0] * (self.Nx[0] - 0.5), self.Nx[0])
        self.ccy = np.linspace(self.left_corner[1] + self.dx[1] * (1 - 0.5),
                               self.left_corner[1] + self.dx[1] * (self.Nx[1] - 0.5), self.Nx[1])
        if self.dim == 2 and self.__xyz_order == {'x': 0, 'y': 2, 'z': 1}:
            self.ccz = self.ccy
        if self.dim == 3:
            self.ccz = np.linspace(self.left_corner[2] + self.dx[2] * (1 - 0.5),
                                   self.left_corner[2] + self.dx[2] * (self.Nx[2] - 0.5), self.Nx[2])

    def __getitem__(self, data_name):
        """ Overload indexing operator [] """

        if data_name in self.data:
            return self.data[data_name].view()
        else:
            if data_name in self.__simplified_names:
                for item in self.__simplified_names[data_name]:
                    if item in self.data:
                        return self.data[item]
            if data_name in self.__simplified_components:
                if self.__simplified_names[data_name[0]][0] in self.data:
                    if self.dim == 2:
                        return self.data[self.__simplified_names[data_name[0]][0]][:, :, self.__xyz_order[data_name[1]]]
                    elif self.dim == 3:
                        return self.data[self.__simplified_names[data_name[0]][0]][:, :, :, self.__xyz_order[data_name[1]]]

            raise KeyError(data_name, " not found. Available are ", ", ".join(self.names))

    def __setitem__(self, data_name, value):
        """ Overload writing access by operator [] """

        raise IOError("Writing access to data is not implemented.")

    def make_ghost_zone(self, data_name, shear_speed, ghost_width):
        """ Make ghost zone for a certain component (using piecewise-linear if with shear).
            The resulting ghost zone differs with Athena's ghost output since Athena use the
            3rd-order Colella & Sekora extremum preserving algorithm (PPME) for reconstruction
            :param data_name: the name of a desired component (must from self.names)
            :param shear_speed: q * Omega * Lx
            :param ghost_width: the number of ghost cells
        """

        assert(ghost_width > 0)
        gw = int(ghost_width)
        if self.ghost_width == 0:
            self.ghost_width = gw
        elif self.ghost_width > 0 and self.ghost_width != gw:
            warnings.warn("It seems previous ghost zones used a different ghost width: " + str(self.ghost_width)
                          + "\nUsing a new width will update the cell-center coordinates ccx/y/z_gh and l/r_corner_gh.")

        self.left_corner_gh = self.left_corner - self.dx * gw
        self.right_corner_gh = self.right_corner + self.dx * gw
        self.ccx_gh = np.linspace(self.left_corner[0] + self.dx[0] * (1 - 0.5 - gw),
                                  self.left_corner[0] + self.dx[0] * (self.Nx[0] - 0.5 + gw), self.Nx[0] + 2 * gw)
        self.ccy_gh = np.linspace(self.left_corner[1] + self.dx[1] * (1 - 0.5 - gw),
                                  self.left_corner[1] + self.dx[1] * (self.Nx[1] - 0.5 + gw), self.Nx[1] + 2 * gw)
        if self.dim == 3:
            self.ccz_gh = np.linspace(self.left_corner[2] + self.dx[2] * (1 - 0.5 - gw),
                                      self.left_corner[2] + self.dx[2] * (self.Nx[2] - 0.5 + gw), self.Nx[2] + 2 * gw)

        tensor_type = self.svtypes[self.names.index(data_name)]

        shear_distance = abs(shear_speed) * self.t
        shear_distance = shear_distance - np.floor(shear_distance/self.box_size[0]) * self.box_size[0]
        shear_distance_in_cells = shear_distance / self.dx[1]
        sheared_cells = int(np.floor(shear_distance_in_cells))
        shear_fraction = 1 - (shear_distance_in_cells - np.floor(shear_distance_in_cells))
        print("shear_distance = ", shear_distance, ", shear_fraction = ",
              shear_fraction, ", sheared_cells = ", sheared_cells)

        if self.dim == 2:
            if tensor_type == "SCALARS":
                self.data_gh[data_name] = np.zeros(np.flipud(self.Nx[:self.dim] + 2 * gw),
                                                   dtype=self.data[data_name].dtype)
                self.data_gh[data_name][gw:-gw, gw:-gw] = self.data[data_name]

                if self.__xyz_order == {'x': 0, 'y': 2, 'z': 1}:  # x-z 2D shearing box
                    self.data_gh[data_name][:,    :gw] = self.data_gh[data_name][:, -2*gw:-gw ]
                    self.data_gh[data_name][:, -gw:  ] = self.data_gh[data_name][:,    gw:2*gw]
                    self.data_gh[data_name][   :gw, :] = self.data_gh[data_name][-2*gw:-gw , :]
                    self.data_gh[data_name][-gw:  , :] = self.data_gh[data_name][   gw:2*gw, :]
                elif self.__xyz_order == {'x': 0, 'y': 1, 'z': 2}:  # x-y 2D shearing box
                    # copy the radial edges first since shear only happens within the non-ghost domain
                    # it is important to use gw:-gw for the first two copies to avoid wrong shear
                    self.data_gh[data_name][gw:-gw, :gw] = \
                        np.roll(self.data_gh[data_name][gw:-gw, -2*gw:-gw], sheared_cells, axis=0) * shear_fraction \
                        + np.roll(self.data_gh[data_name][gw:-gw, -2*gw:-gw], sheared_cells + 1, axis=0) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][gw:-gw, -gw:] = \
                        np.roll(self.data_gh[data_name][gw:-gw, gw:2*gw], -sheared_cells, axis=0) * shear_fraction \
                        + np.roll(self.data_gh[data_name][gw:-gw, gw:2*gw], -sheared_cells - 1, axis=0) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][   :gw, :] = self.data_gh[data_name][-2*gw:-gw , :]
                    self.data_gh[data_name][-gw:  , :] = self.data_gh[data_name][   gw:2*gw, :]
                else:
                    raise NotImplementedError("Making ghost zone for the xyz_order of ", self.__xyz_order,
                                              " has not been implemented")
            if tensor_type == "VECTORS":
                self.data_gh[data_name] = np.zeros(np.hstack([np.flipud(self.Nx[:self.dim] + 2 * gw), 3]),
                                                   dtype=self.data[data_name].dtype)
                self.data_gh[data_name][gw:-gw, gw:-gw, :] = self.data[data_name]

                if self.__xyz_order == {'x': 0, 'y': 2, 'z': 1}:  # x-z 2D shearing box
                    self.data_gh[data_name][:,    :gw, :] = self.data_gh[data_name][:, -2*gw:-gw , :]
                    self.data_gh[data_name][:, -gw:  , :] = self.data_gh[data_name][:,    gw:2*gw, :]
                    self.data_gh[data_name][   :gw, :, :] = self.data_gh[data_name][-2*gw:-gw , :, :]
                    self.data_gh[data_name][-gw:  , :, :] = self.data_gh[data_name][   gw:2*gw, :, :]
                elif self.__xyz_order == {'x': 0, 'y': 1, 'z': 2}:  # x-y 2D shearing box
                    self.data_gh[data_name][gw:-gw, :gw, :] = \
                        np.roll(self.data_gh[data_name][gw:-gw, -2*gw:-gw, :], sheared_cells, axis=0) * shear_fraction \
                        + np.roll(self.data_gh[data_name][gw:-gw, -2*gw:-gw, :], sheared_cells + 1, axis=0) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][gw:-gw, -gw:, :] = \
                        np.roll(self.data_gh[data_name][gw:-gw, gw:2*gw, :], -sheared_cells, axis=0) * shear_fraction \
                        + np.roll(self.data_gh[data_name][gw:-gw, gw:2*gw, :], -sheared_cells - 1, axis=0) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][   :gw, :, :] = self.data_gh[data_name][-2*gw:-gw , :, :]
                    self.data_gh[data_name][-gw:  , :, :] = self.data_gh[data_name][   gw:2*gw, :, :]
                else:
                    raise NotImplementedError("Making ghost zone for the xyz_order of ", self.__xyz_order,
                                              " has not been implemented")
        if self.dim == 3:
            if tensor_type == "SCALARS":
                self.data_gh[data_name] = np.zeros(np.flipud(self.Nx[:self.dim] + 2 * gw),
                                                   dtype=self.data[data_name].dtype)
                self.data_gh[data_name][gw:-gw, gw:-gw, gw:-gw] = self.data[data_name]

                if self.__xyz_order == {'x': 0, 'y': 1, 'z': 2}:  # x-y-z 3D shearing box
                    self.data_gh[data_name][:, gw:-gw, :gw] = \
                        np.roll(self.data_gh[data_name][:, gw:-gw, -2*gw:-gw], sheared_cells, axis=1) * shear_fraction \
                        + np.roll(self.data_gh[data_name][:, gw:-gw, -2*gw:-gw], sheared_cells + 1, axis=1) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][:, gw:-gw, -gw:] = \
                        np.roll(self.data_gh[data_name][:, gw:-gw, gw:2*gw], -sheared_cells, axis=1) * shear_fraction \
                        + np.roll(self.data_gh[data_name][:, gw:-gw, gw:2*gw], -sheared_cells - 1, axis=1) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][:,    :gw, :] = self.data_gh[data_name][:, -2*gw:-gw , :]
                    self.data_gh[data_name][:, -gw:  , :] = self.data_gh[data_name][:,    gw:2*gw, :]
                    self.data_gh[data_name][   :gw, :, :] = self.data_gh[data_name][-2*gw:-gw , :, :]
                    self.data_gh[data_name][-gw:  , :, :] = self.data_gh[data_name][   gw:2*gw, :, :]
                else:
                    raise NotImplementedError("Making ghost zone for the xyz_order of ", self.__xyz_order,
                                              " has not been implemented")
            if tensor_type == "VECTORS":
                self.data_gh[data_name] = np.zeros(np.hstack([np.flipud(self.Nx[:self.dim] + 2 * gw), 3]),
                                                   dtype=self.data[data_name].dtype)
                self.data_gh[data_name][gw:-gw, gw:-gw, gw:-gw, :] = self.data[data_name]

                if self.__xyz_order == {'x': 0, 'y': 1, 'z': 2}:  # x-y-z 3D shearing box
                    self.data_gh[data_name][:, gw:-gw, :gw, :] = \
                        np.roll(self.data_gh[data_name][:, gw:-gw, -2*gw:-gw, :], sheared_cells, axis=1) * shear_fraction \
                        + np.roll(self.data_gh[data_name][:, gw:-gw, -2*gw:-gw, :], sheared_cells + 1, axis=1) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][:, gw:-gw, -gw:, :] = \
                        np.roll(self.data_gh[data_name][:, gw:-gw, gw:2*gw, :], -sheared_cells, axis=1) * shear_fraction \
                        + np.roll(self.data_gh[data_name][:, gw:-gw, gw:2*gw, :], -sheared_cells - 1, axis=1) \
                        * (1 - shear_fraction)
                    self.data_gh[data_name][:,    :gw, :, :] = self.data_gh[data_name][:, -2*gw:-gw , :, :]
                    self.data_gh[data_name][:, -gw:  , :, :] = self.data_gh[data_name][:,    gw:2*gw, :, :]
                    self.data_gh[data_name][   :gw, :, :, :] = self.data_gh[data_name][-2*gw:-gw , :, :, :]
                    self.data_gh[data_name][-gw:  , :, :, :] = self.data_gh[data_name][   gw:2*gw, :, :, :]
                else:
                    raise NotImplementedError("Making ghost zone for the xyz_order of ", self.__xyz_order,
                                              " has not been implemented")

    def plot_slice(self, data, normal='z', slicing=np.s_[:], ax=None, figsize=None,
                   action=None, log_norm=None, **kwargs):
        """
        Plot a 2D slice from 2D or 3D data (assuming the typical xyz_order or 2D xz)
        :param data: the name of a desired component or a 2D numpy ndarray
        :param normal: the normal direction to the slice (e.g., 'z', 'y'; ignored for 2D data)
        :param slicing: data slicing along the normal direction (ignored for 2D data)
        :param ax: Axes object for plotting; will create one if None
        :param figsize: customized figure size
        :param action: a function object, performing action towards data (must return a 2D array)
                       (e.g., lambda x : np.sum(x, axis=1), ignored for 2D data)
                       note that action will be executed after slicing and before log_norm
        :param log_norm: whether or not to plot normalized data in the log-scale
        :param **kwargs: more keywords for ax.pcolorfast (e.g., vmin, vmax)

        to get this Image object for plotting colorbar, use ax.images[0]
        to get image data from ax, use ax.images[0].get_array()
        N.B., values <= 0 will be masked in ax.images[0].get_array()
        """

        new_ax_flag = False
        if ax is None:
            plt_params("medium")
            fig, ax = plt.subplots(figsize=figsize)
            new_ax_flag = True

        if isinstance(data, str):
            if log_norm is None:
                log_norm = True if data in ['rhop', 'dpar', 'particle_density'] else False
            data = self[data]
        elif isinstance(data, (list, tuple, array, np.ndarray)):
            data = np.asarray(data)
            # b/c ccx/y/z is uniform, pcolorfast will create grid based on their bounds, data shape won't matter
            if data.ndim != self.dim:
                raise ValueError("Input data must match the dimension of the original VTK data. Got:", data.ndim)

        if self.dim == 2:
            # ccy is identical to ccz for 2D xz simulations
            if action is not None:
                data = action(data)
            if log_norm: data = np.log10(data / data.mean())
            try:
                ax.pcolormesh(self.ccx, self.ccy, data, shading='auto', **kwargs)
            except:
                ax.pcolorfast(self.ccx, self.ccy, data, **kwargs)
            if new_ax_flag:
                if self.__xyz_order['z'] == 2:
                    ax_labeling(ax, x=r"$x/H$", y=r"$y/H$")
                elif self.__xyz_order['z'] == 1:
                    ax_labeling(ax, x=r"$x/H$", y=r"$z/H$")
        if self.dim == 3:
            # np.s_[x] will directly become an integer
            # np.s_[x:x+1] will give a singleton dimension (preserving ndim)
            if not isinstance(slicing, (int, slice)):
                raise TypeError("slicing must be a slice type, use np.s_[] or integers for this keyword")
            if normal == 'z':
                data = data[slicing, :, :]
                if data.ndim == 3:
                    if action is None:
                        data = data.mean(axis=0)
                    else:
                        data = action(data)
                        if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                        if data.ndim != 2: raise ValueError("The expected ndim of data is 2, but got", data.ndim)
                if log_norm: data = np.log10(data / data.mean())
                try:
                    ax.pcolormesh(self.ccx, self.ccy, data, shading='auto', **kwargs)
                except AttributeError:  # in case matplotlib.__version__ is low where shading='auto' is not there
                    ax.pcolorfast(self.ccx, self.ccy, data, **kwargs)
                if new_ax_flag: ax_labeling(ax, x=r"$x/H$", y=r"$y/H$")
            elif normal == 'y':
                data = data[:, slicing, :]
                if data.ndim == 3:
                    if action is None:
                        data = data.mean(axis=1)
                    else:
                        data = action(data)
                        if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                        if data.ndim != 2: raise ValueError("The expected ndim of data is 2, but got", data.ndim)
                if log_norm: data = np.log10(data / data.mean())
                try:
                    ax.pcolormesh(self.ccx, self.ccy, data, shading='auto', **kwargs)
                except AttributeError:  # in case matplotlib.__version__ is low where shading='auto' is not there
                    ax.pcolorfast(self.ccx, self.ccy, data, **kwargs)
                if new_ax_flag: ax_labeling(ax, x=r"$x/H$", y=r"$z/H$")
            elif normal == 'x':
                data = data[:, :, slicing]
                if data.ndim == 3:
                    if action is None:
                        data = data.mean(axis=2)
                    else:
                        data = action(data)
                        if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                        if data.ndim != 2: raise ValueError("The expected ndim of data is 2, but got", data.ndim)
                if log_norm: data = np.log10(data / data.mean())
                try:
                    ax.pcolormesh(self.ccx, self.ccy, data, shading='auto', **kwargs)
                except AttributeError:  # in case matplotlib.__version__ is low where shading='auto' is not there
                    ax.pcolorfast(self.ccx, self.ccy, data, **kwargs)
                if new_ax_flag: ax_labeling(ax, x=r"$y/H$", y=r"$z/H$")
            else:
                raise ValueError("keyword normal can only be 'z', 'y', or 'x'.")

        if new_ax_flag:
            ax.set_aspect(1.0)
            return fig, ax

    def plot_line(self, data, along='x', slicing=None, ax=None, figsize=None,
                  action=None, **kwargs):
        """
        Plot a 1D line average from 2D or 3D data (assuming the typical xyz_order or 2D xz)
        :param data: the name of a desired component or a 2D numpy ndarray
        :param along: the direction of interest to plot at x-axis
        :param slicing: data slicing perpendicular to the 'along' direction
        :param ax: Axes object for plotting; will create one if None
        :param figsize: customized figure size
        :param action: a function object, performing action towards data (must return a 1D array)
                       (e.g., lambda x : np.sum(x, axis=(0, 1)) )
        :param figsize: customized figure size
        :param **kwargs: more keywords for ax.plot (e.g., lw, alpha)

        to get this Line object, use ax.lines[0]
        to get line data from ax, use ax.lines[0].get_data() or get_xdata() or get_ydata()
        """

        new_ax_flag = False
        if ax is None:
            plt_params("medium")
            fig, ax = plt.subplots(figsize=figsize)
            new_ax_flag = True

        if isinstance(data, str):
            data = self[data]
        elif isinstance(data, (list, tuple, array, np.ndarray)):
            data = np.asarray(data)
            if data.ndim != self.dim:
                raise ValueError("Input data must match the dimension of the original VTK data. Got:", data.ndim)

        if self.dim == 2:
            if slicing is None: slicing = np.s_[:]
            if isinstance(slicing, tuple):
                print("Warning: only the first element of slicing will be used.")
                slicing = slicing[0]
            if isinstance(slicing, (int, slice)):
                if along == 'x':
                    data = data[slicing, :]
                    if data.ndim == 2:
                        if action is None:
                            data = data.mean(axis=0)
                        else:
                            data = action(data)
                            if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                            if data.ndim != 1: raise ValueError("The expected ndim of data is 1, but got", data.ndim)
                    ax.plot(self.ccx, data, **kwargs)
                    if new_ax_flag: ax.set_xlabel(r"$x/H$")
                elif along == 'y' or along == 'z':
                    data = data[:, slicing]
                    if data.ndim == 2:
                        if action is None:
                            data = data.mean(axis=1)
                        else:
                            data = action(data)
                            if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                            if data.ndim != 1: raise ValueError("The expected ndim of data is 1, but got", data.ndim)
                    ax.plot(self.ccy, data, **kwargs)
                    if new_ax_flag: ax.set_xlabel(r"$"+along+r"/H$")
                else:
                    raise ValueError("keyword along can only be 'x', 'y', or 'z'.")
            else:
                raise TypeError("Unexpected slicing for 2D data: slicing=", slicing)
        if self.dim == 3:
            if slicing is None:
                slicing = np.s_[:, :]
            elif isinstance(slicing, (int, slice)):
                print("Warning: duplicating 1D slicing to 2D for plotting 3D data")
                if isinstance(slicing, int):
                    slicing = np.s_[slicing:slicing+1, slicing:slicing+1]
                elif isinstance(slicing, slice):
                    slicing = (slicing, slicing)
            elif isinstance(slicing, tuple):
                if len(slicing) == 0:
                    slicing = np.s_[:, :]
                if len(slicing) == 1:
                    slicing = slicing[0]
                    print("Warning: duplicating 1D slicing to 2D for plotting 3D data")
                    if isinstance(slicing, int):
                        slicing = np.s_[slicing:slicing + 1, slicing:slicing + 1]
                    elif isinstance(slicing, slice):
                        slicing = (slicing, slicing)
                    else:
                        raise TypeError("The type of slicing (element) must be int or slice. type(slicing)=", type(slicing))
                elif len(slicing) >= 2:
                    slicing = list(slicing)
                    for i in range(len(slicing)):
                        if isinstance(slicing[i], int):
                            slicing[i] = np.s_[slicing[i]:slicing[i]+1]
                        elif isinstance(slicing[i], slice):
                            pass
                        else:
                            raise TypeError("The type of slicing element must be int or slice. type(slicing)=", type(slicing[i]))
                    if len(slicing) > 2:
                        print("Warning: only the first two elements of slicing will be used.")
                    slicing = tuple(slicing)

            if along == 'x':
                if action is None:
                    data = data[slicing[0], slicing[1], :].mean(axis=(0, 1))
                else:
                    data = action(data)
                    if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                    if data.ndim != 1: raise ValueError("The expected ndim of data is 1, but got", data.ndim)
                ax.plot(self.ccx, data, **kwargs)
                if new_ax_flag: ax.set_xlabel(r"$x/H$")
            elif along == 'y':
                if action is None:
                    data = data[slicing[0], :, slicing[1]].mean(axis=(0, 2))
                else:
                    data = action(data)
                    if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                    if data.ndim != 1: raise ValueError("The expected ndim of data is 1, but got", data.ndim)
                ax.plot(self.ccy, data, **kwargs)
                if new_ax_flag: ax.set_xlabel(r"$y/H$")
            elif along == 'z':
                if action is None:
                    data = data[:, slicing[0], slicing[1]].mean(axis=(1, 2))
                else:
                    data = action(data)
                    if not isinstance(data, np.ndarray): raise TypeError("Unexpected data type:", type(data))
                    if data.ndim != 1: raise ValueError("The expected ndim of data is 1, but got", data.ndim)
                ax.plot(self.ccz, data, **kwargs)
                if new_ax_flag: ax.set_xlabel(r"$z/H$")
            else:
                raise ValueError("keyword along can only be 'x', 'y', or 'z'.")

        if new_ax_flag:
            return fig, ax

    def map2finer_grid(self, data, finer_shape, orders=(1, 1), nlev=None, return_cc=False):
        """ Map the grid data to polar data
            :param data: str, the name of a desired component to map or a 2D numpy ndarray to map
            :param finer_shape: 2-element int array, the desired shape of the grid to interpolate
            :param orders: interpolation order in x and y direction
            :param nlev: number of levels to refine (e.g, 1 = refine one level) if finer_shape is set to None
            :param return_cc: whether or not to return the finer_ccx and finer_ccy
        """

        if self.dim != 2:
            raise NotImplementedError("This function currently only works for 2D data.")
        if finer_shape is not None:
            if finer_shape[0] <= self.Nx[0] or finer_shape[1] <= self.Nx[1]:
                raise ValueError("polar_shape must be positive integers")
        if finer_shape is None:
            if nlev is None:
                raise ValueError("If finer_shape is set to None, then nlev must be an integer")
            else:
                finer_shape = np.array([self.Nx[0] * 2**nlev, self.Nx[1] * 2**nlev])

        name_list_flag = False
        if isinstance(data, str):
            if data in ['u', 'v', 'w', 'B']:
                raise NotImplementedError("Only scalar field is supported for now.")
            data = self[data]
        elif isinstance(data, (list, tuple, array, np.ndarray)):
            if len(data) > 0 and np.all([isinstance(item, str) for item in data]):
                name_list_flag = True
                if nlev == 0:  # no need to interpolate
                    if return_cc is False:
                        return [self[item] for item in data]
                    else:
                        return self.ccx, self.ccy, [self[item] for item in data]
                for item in data:
                    if item in ['u', 'v', 'w', 'B']:
                        raise NotImplementedError("Only scalar field is supported for now.")
            else:
                data = np.asarray(data)
                if data.shape != self.Nx[:2][::-1]:  # Nx has a reverse order
                    raise ValueError("Input 2D data must match the shape of the original VTK data. Got:", data.shape)
        if nlev == 0:  # no need to interpolate
            if return_cc is False:
                return data
            else:
                return self.ccx, self.ccy, data

        tmp_ccx = np.linspace(self.box_min[0], self.box_max[0], finer_shape[0] + 1)
        tmp_ccy = np.linspace(self.box_min[1], self.box_max[1], finer_shape[1] + 1)
        finer_ccx = (tmp_ccx[1:] + tmp_ccx[:-1]) / 2.0
        finer_ccy = (tmp_ccy[1:] + tmp_ccy[:-1]) / 2.0
        #X, Y = np.meshgrid(finer_ccx, finer_ccy)

        if name_list_flag is False:
            interp_spline = spint.RectBivariateSpline(self.ccy, self.ccx, data, kx=orders[0], ky=orders[1])
            # ev appears to be so much slower than __call__ when the data size is large
            #finer_data = interp_spline.ev(Y, X)
            finer_data = interp_spline(finer_ccy, finer_ccx)

        else:
            finer_data = []
            for item in data:
                interp_spline = spint.RectBivariateSpline(self.ccy, self.ccx, self[item], kx=orders[0], ky=orders[1])
                #finer_data.append(interp_spline.ev(Y, X))
                finer_data.append(interp_spline(finer_ccy, finer_ccx))
        if return_cc:
            return finer_ccx, finer_ccy, finer_data
        else:
            return finer_data

    def generate_polar_grid(self, polar_origin, polar_shape, radius=None):
        """ Generate polar grid along r and theta
            :param polar_origin: 2-element array, the origin of the polar coordinates to map
            :param polar_shape: 2-element int array, the desired shape of the grid to interpolate
            :param radius: radius wanted from the mapping/interpolation
        """
        r_max = min(self.box_max[0] - polar_origin[0], polar_origin[0] - self.box_min[0],
                    self.box_max[1] - polar_origin[1], polar_origin[1] - self.box_min[1])
        if radius is not None:
            r_max = min(r_max, radius)

        cer = np.linspace(0, r_max, polar_shape[0] + 1)  # cell edge in radius
        r = (cer[1:] + cer[:-1]) / 2
        cet = np.linspace(-np.pi, np.pi, polar_shape[1] + 1)  # cell edge in theta
        t = (cet[1:] + cet[:-1]) / 2

        return r, t

    def map2polar(self, data, polar_origin, polar_shape,
                  orders=(1, 1), radius=None):
        """ Map the grid data to polar data
            :param data: str, the name of a desired component to map or a 2D numpy ndarray to map
            :param polar_origin: 2-element array, the origin of the polar coordinates to map
            :param polar_shape: 2-element int array, the desired shape of the grid to interpolate
            :param orders: interpolation order in x and y direction
            :param radius: radius wanted from the mapping/interpolation
        """

        if self.dim != 2:
            raise NotImplementedError("This function currently only works for 2D data.")
        if (polar_origin[0] < self.box_min[0] or polar_origin[0] > self.box_max[0]
            or polar_origin[1] < self.box_min[1] or polar_origin[1] > self.box_max[1]):
            raise NotImplementedError("This function only supports polar_origin within domain")
        if polar_shape[0] <= 0 or polar_shape[1] <= 0:
            raise ValueError("polar_shape must be positive integers")

        name_list_flag = False
        name_list = None
        if isinstance(data, str):
            if data in ['u', 'v', 'w', 'B']:
                raise NotImplementedError("Only scalar field is supported for now.")
            name_list = [data]
            data = self[data]
        elif isinstance(data, (list, tuple, array, np.ndarray)):
            if len(data) > 0 and np.all([isinstance(item, str) for item in data]):
                name_list_flag = True
                for item in data:
                    if item in ['u', 'v', 'w', 'B']:
                        raise NotImplementedError("Only scalar field is supported for now.")
                name_list = data
            else:
                data = np.asarray(data)
                if np.any(data.shape != self.Nx[:2][::-1]):  # Nx has a reverse order
                    raise ValueError("Input 2D data must match the shape of the original VTK data. Got:", data.shape)

        r_max = min(self.box_max[0] - polar_origin[0], polar_origin[0] - self.box_min[0],
                    self.box_max[1] - polar_origin[1], polar_origin[1] - self.box_min[1])
        if radius is not None:
            r_max = min(r_max, radius)

        cer = np.linspace(0, r_max, polar_shape[0] + 1)  # cell edge in radius
        r = (cer[1:] + cer[:-1]) / 2
        cet = np.linspace(-np.pi, np.pi, polar_shape[1] + 1) # cell edge in theta
        t = (cet[1:] + cet[:-1]) / 2

        if name_list_flag is True:
            polar_map = SimpleMap2Polar2D(self.ccx, self.ccy, [self[item] for item in data], r, t,
                                          origin=polar_origin, orders=orders, data_names=name_list)
        else:
            polar_map = SimpleMap2Polar2D(self.ccx, self.ccy, data, r, t,
                                          origin=polar_origin, orders=orders, data_names=name_list)

        return polar_map

    def incorporate_trimmed_par_data(self, par, shapeshifting=True):
        """ After applying split_VTK_and_trim_par() to reduce storage space, one may hope to re-construct
            the particle data ndarray into the original shape from the trimmed data set
            :param par: another AthenaVTK instance that holds the trimmed particle data
            :param shapeshifting: whether to update data in-place and update time/dt
        """

        if not isinstance(par, AthenaVTK):
            raise TypeError("The input parameter is not an instance of AthenaVTK.")
        try:
            idz_min, idz_max = np.argmin(np.abs(par.ccz[0] - self.ccz)), np.argmin(np.abs(par.ccz[-1] - self.ccz))
        except AttributeError:
            # if ccz not working, it is likely the data is in 2D
            idz_min, idz_max = np.argmin(np.abs(par.ccy[0] - self.ccy)), np.argmin(np.abs(par.ccy[-1] - self.ccy))
        except:
            raise RuntimeError("Cannot access either ccz or ccy. Please check data.")

        if shapeshifting:
            # update data in place
            self.t, self.level, self.domain = par.t, par.level, par.domain

        shape = self.Nx[::-1]
        if shape[0] == 0:  # for 2D data
            shape = shape[1:]
        common_types = {'float': np.float32, 'double': np.float64, 'int': int}
        for idx, _name in enumerate(par.names):
            if par.dtypes[idx] not in common_types:
                raise TypeError(f"Unrecognized type: {par.dtypes[idx]}")
            if _name in self.names:
                if shapeshifting is not True:
                    raise RuntimeError(f"Data '{_name}' will be lost if continue." + "\nAbort now...")
                self.svtypes[self.names.index(_name)] = par.svtypes[idx]
                self.dtypes[self.names.index(_name)] = par.dtypes[idx]
            else:
                self.names.append(_name)
                self.svtypes.append(par.svtypes[idx])
                self.dtypes.append(par.dtypes[idx])

            if par.svtypes[idx] == 'SCALARS':
                self.data[_name] = np.zeros(shape, dtype=common_types[par.dtypes[idx]])
                self.data[_name][idz_min:idz_max+1] = par[_name]
            elif par.svtypes[idx] == 'VECTORS':
                self.data[_name] = np.zeros(np.hstack([shape, 3]), dtype=common_types[par.dtypes[idx]])
                self.data[_name][idz_min:idz_max+1] = par[_name]
            else:
                raise NotImplementedError("Dataset attribute " + par.svtypes[idx] + "not supported.")


def split_VTK_and_trim_par(filename, par_filename, gas_filename, **kwargs):
    """ Split a VTK data file into two files, one contains vertically trimmed "particle_density"
        and "particle_momentum" (since particles are usually concentrated near the midplane);
        The other one contains other data (unmodified).
        :param filename: the file name of the original VTK file to read
        :param par_filename: the output file for trimmed particle data (i.e., rho_p + v)
        :param gas_filename: the output file for other unmodified data
    """

    read_kwargs = kwargs.get('read_kw', {})
    a = AthenaVTK(filename, **read_kwargs)

    base_q = kwargs.get('base_q', 'particle_density')
    if len(a.names) == 1:
        base_q = a.names[0]
    if base_q not in a.data:
        _q = None
        if base_q in a._simplified_names:
            for item in a._simplified_names[base_q]:  # returns a list
                if item in a.data:
                    _q = item
                    break
        if _q is None:
            raise ValueError("Cannot find '"+base_q+"' in the dataset. Available: ", a.names)
        # no need to replace base_q with full name

    trim_q = kwargs.get('trim_q', ['particle_density', 'particle_momentum'])
    if len(a.names) == 1:
        trim_q = [a.names[0]]
    if isinstance(trim_q, str):
        trim_q = [trim_q]
    q2trim = []  # we need full names below to determine whether to trim or not
    for _q in trim_q:
        if _q in a.data:
            q2trim.append(_q)
        else:
            _qq = None
            if _q in a._simplified_names:
                for item in a._simplified_names[_q]:  # returns a list
                    if item in a.data:
                        _qq = item
                        break
            if _qq is None:
                raise ValueError("Cannot find '"+_q+"' in the dataset to trim. Available: ", a.names)
            else:
                q2trim.append(_qq)

    only_split_par_flag = kwargs.get("only_split_par", False)
    if set(a.names).issubset(set(q2trim)):
        only_split_par_flag = True

    # after this, we can assume it is either 3D or 2D
    if a.dim == 3:
        print(f"3D data, looks like Nz = Nx[2] = {a.Nx[2]}")
        Nz = a.Nx[2]
        if Nz <= 1:
            raise ValueError("Although the data seems to be 3D, Nz={Nz}. Please check the data...")
    elif a.dim == 2:
        print(f"2D data, looks like Nz = Nx[1] = {a.Nx[1]}")
        Nz = a.Nx[1]
        if Nz <= 1:
            raise ValueError("Although the data seems to be 2D, Nz={Nz}. Please check the data...")
    else:
        raise NotImplementedError("This function currently only support 2D/3D data.")

    idz_min, idz_max = -1, Nz
    for idz in range(Nz // 2):
        if np.all(a[base_q][idz] == 0):
            idz_min = idz
        else:
            break
    for idz in range(Nz - 1, Nz // 2 - 1, -1):
        if np.all(a[base_q][idz] == 0):
            idz_max = idz
        else:
            break

    if idz_min == -1 and idz_max == Nz:
        raise ValueError("The dataset 'particle_density' has non-zero values at all height (cannot be trimmed).")

    only_split_gas_flag = kwargs.get("only_split_gas", False)
    if idz_min == Nz // 2 - 1 and idz_max == Nz // 2:
        print("The dataset 'particle_density' are all zeros.")
        if not only_split_gas_flag or only_split_par_flag:
            return None

    idz_min += 1
    idz_max -= 1  # the min and max indices that have non-zero values
    par_Nz = (idz_max + 1 - idz_min)
    if a.dim == 3:
        par_size = a.Nx[0] * a.Nx[1] * par_Nz
    else:
        par_size = a.Nx[0] * par_Nz
    print(f"idz = {idz_min}:{idz_max}, par_Nz = {par_Nz}, par_size = {par_size}")
    if only_split_par_flag:
        print("only_split_par_flag is set to True")
    if only_split_gas_flag:
        print("only_split_gas_flag is set to True")
    if only_split_par_flag and only_split_gas_flag:
        print("Only_split_par_flag and only_split_gas_flag are both set to True. quit...")
        return None

    fo = open(filename, "rb")  # original data file
    eof = fo.seek(0, 2)  # record eof position
    fo.seek(0, 0)
    if not only_split_par_flag:
        fg = open(gas_filename, "wb")  # new data file for other (mostly gas) quantities
    if not only_split_gas_flag:
        fp = open(par_filename, "wb")  # new data file for trim-mable particle quantities

    for l in range(4):
        # from version comment to DATASET UNSTRUCTURED_POINTS
        tmp_line = fo.readline()
        if not only_split_par_flag:
            fg.write(tmp_line)
        if not only_split_gas_flag:
            fp.write(tmp_line)
    # DIMENSIONS X, Y, Z
    tmp_line = fo.readline()
    if not only_split_par_flag:
        fg.write(tmp_line)
    if not only_split_gas_flag:
        if a.dim == 3:
            fp.write(f"DIMENSIONS {a.Nx[0] + 1} {a.Nx[1] + 1} {par_Nz + 1}\n".encode('utf-8'))
        else:
            fp.write(f"DIMENSIONS {a.Nx[0] + 1} {par_Nz + 1} 1\n".encode('utf-8'))
    # ORIGIN %e %e %e
    tmp_line = fo.readline()
    if not only_split_par_flag:
        fg.write(tmp_line)
    if not only_split_gas_flag:
        if a.dim == 3:
            fp.write(f"ORIGIN {a.box_min[0]:e} {a.box_min[1]:e} {a.ccz[idz_min] - a.dx[2] / 2:e}\n".encode('utf-8'))
        else:
            try:
                fp.write(f"ORIGIN {a.box_min[0]:e} {a.ccz[idz_min] - a.dx[1] / 2:e} {a.box_min[2]:e}\n".encode('utf-8'))
            except AttributeError:
                fp.write(f"ORIGIN {a.box_min[0]:e} {a.ccy[idz_min] - a.dx[1] / 2:e} {a.box_min[2]:e}\n".encode('utf-8'))
            except:
                raise ValueError("Cannot access ccz or ccy in 2D data")
    # SPACING %e %e %e
    tmp_line = fo.readline()
    if not only_split_par_flag:
        fg.write(tmp_line)
    if not only_split_gas_flag:
        fp.write(tmp_line)
    # CELL_DATA N_size
    tmp_line = fo.readline()
    if not only_split_par_flag:
        fg.write(tmp_line)
    if not only_split_gas_flag:
        fp.write(f"CELL_DATA {par_size}\n".encode('utf-8'))

    common_types = {'float': 'f', 'double': 'd', 'int': 'i'}
    while fo.tell() != eof:
        _tmp_line = fo.readline().decode('utf-8')
        if _tmp_line == '\n':  # just in case it is an empty line
            _tmp_line = fo.readline().decode('utf-8')
        tmp_line = _tmp_line.split()
        print("now processing", tmp_line)

        try:
            type_char = common_types[tmp_line[2]]
        except KeyError:
            warnings.warn("Unrecognized type: " + tmp_line[2] + " for data named "
                          + tmp_line[1] + ". Use float for now.")
            type_char = 'f'

        f2write = None
        if tmp_line[1] in q2trim:
            if not only_split_gas_flag:
                f2write = fp
        else:
            if not only_split_par_flag:
                f2write = fg
        if f2write is None:
            if tmp_line[0] == 'SCALARS':
                fo.readline()
                fo.seek(a.size * struct.calcsize(type_char), 1)
            elif tmp_line[0] == 'VECTORS':
                fo.seek(3 * a.size * struct.calcsize(type_char), 1)
            else:
                raise RuntimeError("The code shouldn't end up here. Report a bug please.")
            continue

        f2write.write(_tmp_line.encode('utf-8'))
        if tmp_line[0] == "SCALARS":
            f2write.write(fo.readline())  # "LOOKUP_TABLE default"

            if tmp_line[1] in q2trim:
                f2write.write((a[tmp_line[1]][idz_min:idz_max + 1]).flatten().byteswap().tobytes())
                fo.seek(a.size * struct.calcsize(type_char), 1)
            else:
                tmp_data = array(type_char)
                tmp_data.fromfile(fo, a.size)
                f2write.write(tmp_data)
            # f2write.write('\n'.encode('utf-8')) # debug use
        elif tmp_line[0] == "VECTORS":
            if tmp_line[1] in q2trim:
                f2write.write((a[tmp_line[1]][idz_min:idz_max + 1]).flatten().byteswap().tobytes())
                fo.seek(3 * a.size * struct.calcsize(type_char), 1)
            else:
                tmp_data = array(type_char)
                tmp_data.fromfile(fo, a.size * 3)  # even for 2D simulations, the vector fields are 3D
                f2write.write(tmp_data)
        else:
            raise NotImplementedError("Dataset attribute " + tmp_line[0] + " not supported.")

    fo.close()
    if not only_split_par_flag:
        fg.close()
    if not only_split_gas_flag:
        fp.close()


def trim_VTK(filename, out_filename, **kwargs):
    """ Trim VTK file size by converting 32fp to 24fp, also support float<=>double
        :param filename: the file name of the original VTK file to read
        :param out_filename: the output file for trimmed VTK data
    """

    allowed_conversions = {  # Define allowed conversions as a set of tuples
        ("float", "float24"),
        ("double", "float"),
        ("float", "double"),
        ("double", "float24")}

    read_kwargs = kwargs.get('read_kw', {})
    ds = AthenaVTK(filename, **read_kwargs)  # ds = dataset
    from_type = kwargs.get("from_type", "float")
    to_type = kwargs.get("to_type", "float24")
    if (from_type, to_type) not in allowed_conversions:
        raise NotImplementedError(f"Conversion from {from_type} to {to_type} not supported.")

    if "trim_q" not in kwargs:
        trim_q = []
        for idq in range(len(ds.names)):
            if ds.dtypes[idq] == from_type:
                trim_q.append(ds.names[idq])
        q2trim = trim_q
    else:
        trim_q = kwargs.get("trim_q")
        if isinstance(trim_q, str):
            trim_q = [trim_q]
        q2trim = []  # we need full names below to determine whether to trim or not
        for _q in trim_q:
            if _q in ds.names and ds.dtypes[ds.names.index(_q)] == from_type:
                q2trim.append(_q)
            else:
                _qq = None
                if _q in ds._simplified_names:
                    for item in ds._simplified_names[_q]:  # returns a list
                        if item in ds.names and ds.dtypes[ds.names.index(_q)] == from_type:
                            _qq = item
                            break
                if _qq is None:
                    raise ValueError(f"Cannot find {_q} with {from_type} in the dataset to trim. Available: ",
                                     list(zip(ds.names, ds.dtypes)))
                else:
                    q2trim.append(_qq)
    print("q2trim:", q2trim)

    f = open(filename, "rb")  # original data file
    eof = f.seek(0, 2)  # record eof position
    f.seek(0, 0)
    ft = open(out_filename, "wb")  # new data file for trimmed data

    for l in range(4):
        # from version comment to DATASET UNSTRUCTURED_POINTS
        tmp_line = f.readline()
        ft.write(tmp_line)
    # DIMENSIONS X, Y, Z
    tmp_line = f.readline()
    ft.write(tmp_line)
    # ORIGIN %e %e %e
    tmp_line = f.readline()
    ft.write(tmp_line)
    # SPACING %e %e %e
    tmp_line = f.readline()
    ft.write(tmp_line)
    # CELL_DATA N_size
    tmp_line = f.readline()
    ft.write(tmp_line)

    conversion_map = {
        ("float", "float24"): lambda x: convert_array_float32_to_float24(x.flatten()).flatten(),
        ("double", "float"): lambda x: x.flatten().astype(np.float32),
        ("float", "double"): lambda x: x.flatten().astype(np.float64),
        ("double", "float24"): lambda x: convert_array_float32_to_float24(x.flatten().astype(np.float32)).flatten()
    }
    common_types = {'float': 'f', 'double': 'd', 'int': 'i', 'unsigned_char': 'B'}
    while f.tell() != eof:
        _tmp_line = f.readline().decode('utf-8')  # e.g., "SCALARS density float" or "VECTORS momentum float"
        if _tmp_line == '\n':  # just in case it is an empty line
            _tmp_line = f.readline().decode('utf-8')
        tmp_line = _tmp_line.split()
        print("now processing", tmp_line)
        try:
            type_char = common_types[tmp_line[2]]
        except KeyError:
            raise TypeError(f"Unrecognized type: {tmp_line[2]} for data named {tmp_line[1]}.")

        if tmp_line[2] == from_type:
            #print(f"now trim {_tmp_line}")
            if tmp_line[0] == "SCALARS":
                if to_type == "float24":
                    ft.write(f"SCALARS {tmp_line[1]}_fp24 unsigned_char\n".encode())
                else:
                    ft.write(f"SCALARS {tmp_line[1]} {to_type}\n".encode())
                ft.write(f.readline())  # "LOOKUP_TABLE default"
                _dim2write = 1
            elif tmp_line[0] == "VECTORS":
                if to_type == "float24":
                    ft.write(f"VECTORS {tmp_line[1]}_fp24 unsigned_char\n".encode())
                else:
                    ft.write(f"VECTORS {tmp_line[1]} {to_type}\n".encode())
                _dim2write = 3
            else:
                raise NotImplementedError("Dataset attribute " + tmp_line[0] + " not supported.")

            try:
                tmp_data = conversion_map[(from_type, to_type)](ds[tmp_line[1]])
            except KeyError:
                raise NotImplementedError(f"Conversion from {from_type} to {to_type} not supported.")
            except Exception as e:
                raise RuntimeError("Something went wrong during the conversion process.") from e

            ft.write(tmp_data.byteswap().tobytes())
            f.seek(ds.size * struct.calcsize(type_char) * _dim2write, 1)
        else:
            print(f"now preserve {_tmp_line}")
            ft.write(_tmp_line.encode('utf-8'))  # e.g., "SCALARS density float" or "VECTORS momentum float"
            if tmp_line[0] == "SCALARS":
                ft.write(f.readline())  # "LOOKUP_TABLE default"
                _dim2write = 1
            elif tmp_line[0] == "VECTORS":
                _dim2write = 3
            else:
                raise NotImplementedError("Dataset attribute " + tmp_line[0] + " not supported.")
            tmp_data = array(type_char)
            tmp_data.fromfile(f, ds.size * _dim2write)
            ft.write(tmp_data)

    f.close()
    ft.close()
    print(f"New data saved to {out_filename}")

class AthenaMultiVTK(AthenaVTK):
    """ Read data from sub-VTK files from all processors from SI simulations (by Athena)
        AthenaMultiVTK is able to read BINARY data of STRUCTURED_POINTS, either 2D or 3D
        e.g.,
            >>> a = AthenaMultiVTK("bin", "Par_Strat3d", "0001.vtk", silent=False)
            Read [density, momentum, particle_density, particle_momentum] at Nx=[64, 64, 64]

    """

    def __init__(self, data_dir, prefix, postfix, wanted=None, xyz_order=None, silent=True, lev=None):

        id_folders = [x for x in os.listdir(data_dir) if x[:2] == 'id']
        if len(id_folders) == 0:
            raise RuntimeError("No data files to read (no id*)")

        self.num_cpus = len(id_folders)

        if data_dir[-1] != '/':
            data_dir = data_dir + '/'

        if lev is not None:
            if isinstance(lev, str):
                lev = int(lev)
            if isinstance(lev, Number):
                filenames = [data_dir+"id"+str(x)+"/lev"+str(lev)+'/'+prefix+"-id"+str(x)+"-lev"+str(lev)+'.'+postfix for x in range(self.num_cpus)]
                filenames[0] = data_dir+"id0/lev"+str(lev)+'/'+prefix+"-lev"+str(lev)+'.'+postfix
            else:
                raise ValueError("Cannot understand the input level info: ", lev)
        else:
            filenames = [data_dir+"id"+str(x)+'/'+prefix+"-id"+str(x)+'.'+postfix for x in range(self.num_cpus)]
            filenames[0] = data_dir+"id0/"+prefix+'.'+postfix

        super().__init__(filenames[0], wanted=wanted, xyz_order=xyz_order)

        tmp_data = [AthenaVTK(idx, wanted=wanted, xyz_order=xyz_order) for idx in filenames]

        self.origin = self.left_corner[:self.dim]
        self.ending = self.right_corner[:self.dim]
        division_safe_dx = self.dx[:self.dim]
        self.num_cells = 0  # reset to zero for accumulation

        for item in tmp_data:
            self.origin = np.minimum(self.origin, item.left_corner[:self.dim])
            if not np.array_equal(self.dx, item.dx):
                raise RuntimeError("different spacing encountered, diff = ", ["{:.8e}".format(x) for x in  self.dx-item.dx])
            self.num_cells += item.size
            self.ending = np.maximum(self.ending, item.right_corner[:self.dim])
        self.box_min = np.zeros(3)
        self.box_max = np.zeros(3)
        self.box_min[:self.dim] = self.origin
        self.box_max[:self.dim] = self.ending

        self.Nx = np.round((self.ending - self.origin) / division_safe_dx)
        self.Nx = self.Nx.astype(int)
        if np.prod(self.Nx[:self.dim]) != self.num_cells:
            raise RuntimeError("Numbers don't match, Nx=", self.Nx, ", # of cells=", self.num_cells)

        for i in range(len(self.names)):
            if self.svtypes[i] == "SCALARS":
                self.data[self.names[i]] = np.zeros(np.flipud(self.Nx[:self.dim]))
            elif self.svtypes[i] == "VECTORS":
                self.data[self.names[i]] = np.zeros(np.hstack([np.flipud(self.Nx[:self.dim]), 3]))
            else:
                raise RuntimeError("Unknown type: ", self.svtypes[i])

        for item in tmp_data:
            tmp_origin_idx = np.round((item.left_corner[:self.dim] - self.origin) / division_safe_dx)
            tmp_origin_idx = tmp_origin_idx.astype(int)
            tmp_ending_idx = np.round((item.right_corner[:self.dim] - self.origin) / division_safe_dx)
            tmp_ending_idx = tmp_ending_idx.astype(int)

            for i in range(len(self.names)):
                if self.svtypes[i] == "SCALARS":
                    if self.dim == 2:
                        self.data[self.names[i]][tmp_origin_idx[1]:tmp_ending_idx[1], tmp_origin_idx[0]:tmp_ending_idx[0]] = item.data[self.names[i]]
                    elif self.dim == 3:
                        self.data[self.names[i]][tmp_origin_idx[2]:tmp_ending_idx[2], tmp_origin_idx[1]:tmp_ending_idx[1], tmp_origin_idx[0]:tmp_ending_idx[0]] = item.data[self.names[i]]
                elif self.svtypes[i] == "VECTORS":
                    if self.dim == 2:
                        self.data[self.names[i]][tmp_origin_idx[1]:tmp_ending_idx[1], tmp_origin_idx[0]:tmp_ending_idx[0], :] = item.data[self.names[i]]
                    elif self.dim == 3:
                        self.data[self.names[i]][tmp_origin_idx[2]:tmp_ending_idx[2], tmp_origin_idx[1]:tmp_ending_idx[1], tmp_origin_idx[0]:tmp_ending_idx[0], :] = item.data[self.names[i]]

        self._set_cell_centers_()
        self.left_corner[:self.dim] = self.origin
        self.right_corner[:self.dim] = self.ending
        self.size = self.num_cells
        self.box_size[:self.dim] = self.dx[:self.dim] * self.Nx[:self.dim]

        if not silent:
            print("Read [" + ", ".join(self.names) + "] at Nx=[" + ", ".join([str(x) for x in self.Nx]) + "]")


class AthenaSMRVTK:
    """ Read multi-level data from VTK files produced by Athena simulations with Static Mesh Refinement

        ASSUMPTIONS: each level only contains one domain/grid
    """
    def __init__(self, data_dir, problem_id, postfix, nlev=1, serial=False,
                 multi=False, wanted=None, xyz_order=None, silent=True, **kwargs):

        self.data = []
        read_kwargs = {"wanted": wanted, "xyz_order": xyz_order, "silent": silent}
        if serial is True:
            self.data.append(AthenaVTK(data_dir + '/' + problem_id + '.' + postfix, **read_kwargs))
            if nlev > 1:
                for idx_lev in range(1, nlev):
                    self.data.append(AthenaVTK(data_dir + "/lev{}".format(idx_lev) + '/' + problem_id
                                               + "-lev{}.".format(idx_lev) + postfix,
                                               **read_kwargs))
        else:
            if multi is False:
                self.data.append(AthenaVTK(data_dir+'/'+problem_id+'.'+postfix, **read_kwargs))
                if nlev > 1:
                    for idx_lev in range(1, nlev):
                        self.data.append(AthenaVTK(data_dir+'/'+problem_id+"-lev{}.".format(idx_lev)+postfix,
                                                   **read_kwargs))
            else:
                self.data.append(AthenaMultiVTK(data_dir, problem_id, postfix, **read_kwargs))
                if nlev > 1:
                    for idx_lev in range(1, nlev):
                        self.data.append(AthenaMultiVTK(data_dir, problem_id, postfix, lev=idx_lev, **read_kwargs))

        self.num_lev = len(self.data)
        self.dim = self.data[0].dim
        self.box_min = self.data[0].box_min
        self.box_max = self.data[0].box_max
        self.names = self.data[0].names
        self.t = self.data[0].t

        # mesh_ccx and mesh_ccy
        self.meshX = []
        self.meshY = []
        # stacked meshX and meshY
        self.meshXY = []
        # mesh_indices and mesh_masks for every level
        # True         and 1          for effective, non-overlapping regions
        self.mesh_idx = []
        self.mesh_masks = []
        self.generate_level_mask()

        # finest data
        self.finest_ccx = None
        self.finest_ccy = None
        self.finest_meshXY = None
        #self._finest_data = None
        self.finest_data = None
        self._data_dict = None
        self.finest_names = None
        self.lev_mapped = None

        # polar finest data
        self.pd = None  # polar data object
        # mesh array of area fractions of cells outside polar grid, useful when analyzing data in both grids
        self.polar_fraction = None
        self.non_polar_fraction = None
        # a lambda function to get the coordinates of four vertices of a cell
        self.cell_vertices = lambda xy, hdx: [(xy[0] - hdx, xy[1] - hdx), (xy[0] - hdx, xy[1] + hdx),
                                              (xy[0] + hdx, xy[1] + hdx), (xy[0] + hdx, xy[1] - hdx)]

    def __getitem__(self, index):
        """ Overload indexing operator [] """

        if index >= len(self.data):
            raise IndexError("Bad outbound access (overflow); ", index, ">=", len(self.data))
        return self.data[index]

    def plot_domains(self, ax = None, figsize=None, **kwargs):
        """ Plot to show Mesh Refinement domains """

        if self.dim != 2:
            print("Warning: This function currently only consider X-Y plane.")
        new_ax_flag = False
        if ax is None:
            plt_params("medium")
            fig, ax = plt.subplots(figsize=figsize)
            new_ax_flag = True

        for b in self.data:
            ax.plot([b.box_min[0], b.box_min[0]], [b.box_min[1], b.box_max[1]], lw=2, alpha=0.75)
            ax.plot([b.box_max[0], b.box_max[0]], [b.box_min[1], b.box_max[1]], lw=2, alpha=0.75,
                    color=ax.lines[-1].get_color())
            ax.plot([b.box_min[0], b.box_max[0]], [b.box_min[1], b.box_min[1]], lw=2, alpha=0.75,
                    color=ax.lines[-1].get_color())
            ax.plot([b.box_min[0], b.box_max[0]], [b.box_max[1], b.box_max[1]], lw=2, alpha=0.75,
                    color=ax.lines[-1].get_color())

        ax.set_xscale('symlog', linthresh=kwargs.get("linthresh", 0.1))
        ax.set_yscale('symlog', linthresh=kwargs.get("linthresh", 0.1))
        ax.grid(True, ls=':')

        if new_ax_flag:
            ax.set(aspect=kwargs.get("aspect", 1.0))
            return fig, ax

    def generate_level_mask(self):
        """ Generate masks to mark effective/non-overlapping regions at each level """

        if self.dim != 2:
            raise NotImplementedError("This function currently only works for 2D data.")

        # reset if needed
        if len(self.mesh_masks) > 0:
            # mesh_ccx and mesh_ccy
            self.meshX = []
            self.meshY = []
            # stacked meshX and meshY
            self.meshXY = []
            # mesh_indices and mesh_masks for every level
            # True         and 1          for effective, non-overlapping regions
            self.mesh_idx = []
            self.mesh_masks = []

        for l in range(self.num_lev):
            tmp_X, tmp_Y = np.meshgrid(self[l].ccx, self[l].ccy)
            self.meshX.append(tmp_X)
            self.meshY.append(tmp_Y)
            self.meshXY.append(np.dstack([tmp_X, tmp_Y]))

        # set True and 1 for all effective/non-overlapping regions at each level
        for l in range(self.num_lev - 1):
            #tmp_C = ~((self.meshX[l] > self[l+1].box_min[0]) & (self.meshX[l] < self[l+1].box_max[0])
            #          & (self.meshY[l] > self[l+1].box_min[1]) & (self.meshY[l] < self[l+1].box_max[1]))
            tmp_C = ((self.meshX[l] < self[l+1].box_min[0]) | (self.meshX[l] > self[l+1].box_max[0])
                     | (self.meshY[l] < self[l+1].box_min[1]) | (self.meshY[l] > self[l+1].box_max[1]))
            self.mesh_idx.append(tmp_C)
            self.mesh_masks.append(np.zeros_like(self.mesh_idx[l], dtype=int))
            self.mesh_masks[l][self.mesh_idx[l]] = 1

        # 1 for all in the finest level
        self.mesh_masks.append(np.ones_like(self.meshX[-1], dtype=int))

    def map2finest_grid(self, data, lev2map=None, orders=[1, 1]):
        """ Map desired data from selected levels to the finest grid """

        if self.dim != 2:
            raise NotImplementedError("This function currently only works for 2D data.")

        name_list_flag = False
        if isinstance(data, (list, tuple, array, np.ndarray)):
            if len(data) > 0 and np.all([isinstance(item, str) for item in data]):
                name_list_flag = True
                self.finest_names = data
            else:
                raise TypeError("only data names are accepted.")
        elif isinstance(data, str):
            self.finest_names = [data]
        else:
            raise TypeError("only data names are accepted.")

        if lev2map is None:
            lev0 = 0
            lev2map = np.arange(lev0, self.num_lev)
        else:
            lev2map = np.sort(np.atleast_1d(lev2map))  # lowest number first, works even if lev2map is int
            for idx, l in enumerate(lev2map):
                if l < 0:
                    lev2map[idx] = self.num_lev + l
            lev2map = np.sort(lev2map)  # lowest number first, works even if lev2map is int
            lev0 = min(lev2map)

        l = lev2map[0]
        self.finest_ccx, self.finest_ccy, self.finest_data \
             = self[l].map2finer_grid(data, None, orders=orders, nlev=self.num_lev - 1 - self[l].level, return_cc=True)
        self.finest_Nx = np.array([self.finest_ccx.size, self.finest_ccy.size, 0])

        if lev2map.size > 1:
            tmp_X, tmp_Y = np.meshgrid(self.finest_ccx, self.finest_ccy)
            #self._finest_data = [self.finest_data]
            for idx, l in enumerate(lev2map[1:]):
                tmp_data = self[l].map2finer_grid(data, None, orders=orders, nlev=self.num_lev - 1 - self[l].level)
                tmp_C = ~((tmp_X < self[l].box_min[0]) | (tmp_X > self[l].box_max[0])
                          | (tmp_Y < self[l].box_min[1]) | (tmp_Y > self[l].box_max[1]))
                if np.count_nonzero(tmp_C) != self[l].num_cells * (2**(self.num_lev - 1 - self[l].level))**self.dim:
                    raise ValueError("Number of cells mismatch while filling up finest_data")
                if name_list_flag is True:
                    for idx, item in enumerate(data):
                        self.finest_data[idx][tmp_C] = tmp_data[idx].flatten()
                else:
                    # tried to use [tmp_C_y, :][:, tmp_C_x], which seems to be a copy so assignment failed
                    self.finest_data[tmp_C] = tmp_data.flatten()
                #self._finest_data.append(tmp_data)

        self.lev_mapped = lev2map
        self._data_dict = dict()
        #if len(self.finest_names) == 1:
        #    self._data_dict[self.finest_names[0]] = self.finest_data[0].view()
        #else:
        for idx, item in enumerate(self.finest_names):
            self._data_dict[item] = self.finest_data[idx].view()
        self.finest_meshXY = np.dstack([*np.meshgrid(self.finest_ccx, self.finest_ccy)])

    def map_finest2polar(self, polar_origin, polar_shape=None, radius=None, orders=[1, 1]):
        """ map the processed finest data to polar grid
            thus the components being mapped here depends on what map2finest_grid has done
        """

        if self.dim != 2:
            raise NotImplementedError("This function currently only works for 2D data.")
        if self.finest_data is None:
            raise ValueError("Use map2finest_grid to construct finest_data before calling this function.")
        if polar_shape is None:
            polar_shape = [min(self.finest_ccx.size, self.finest_ccy.size)] * 2
        else:
            if polar_shape[0] <= 0 or polar_shape[1] <= 0:
                raise ValueError("polar_shape must be positive integers")
        polar_origin = np.asarray(polar_origin).flatten()
        if polar_origin.size != 2:
            raise ValueError("polar_origin must be a 2-element array/list/tuple")
        if (polar_origin[0] < self[self.lev_mapped[0]].box_min[0]
                or polar_origin[0] > self[self.lev_mapped[0]].box_max[0]
                or polar_origin[1] < self[self.lev_mapped[0]].box_min[1]
                or polar_origin[1] > self[self.lev_mapped[0]].box_max[1]):
            raise NotImplementedError("This function only supports polar_origin within domain")

        r, t = self[self.lev_mapped[0]].generate_polar_grid(polar_origin, polar_shape, radius=radius)

        self.pd = SimpleMap2Polar2D(self.finest_ccx, self.finest_ccy, self.finest_data,
                                    r, t, origin=polar_origin, data_names=self.finest_names, orders=orders)

        # now we construct a 2D array to hold the area fraction of Cartesian cells outside the polar grid
        # i.e., cells fully outside (inside) the polar grid has 1 (0), otherwise, it is between 0 and 1

        self.finest_polar_frac(self.pd.r_max, polar_origin)

    def _dist2origin(self, ccx, ccy, origin):
        """ Calculate distance from cell to origin """

        coords_XY = np.dstack([*np.meshgrid(ccx, ccy)])
        dist2origin = ((coords_XY - origin) ** 2).sum(axis=2) ** 0.5
        return dist2origin

    def finest_polar_frac(self, radius, origin, shapely_res=64):
        """ Calculate fractions of Cartesian cells' area inside/outside a circle
            :param radius: float, radius of the circle
            :param origin: 2-element array, the origin of the circle
            :param shapely_res: int, resolution for point buffer object by shapely,
                                64 => geometric area error ~ 1e-4
                                16 => geometric area error ~ 1.6e-3
        """

        dist2origin = ((self.finest_meshXY - origin) ** 2).sum(axis=2) ** 0.5
        half_dx = self[-1].dx[0] / 2  # !!! BIG ASSUMPTION !!!: square cell
        half_diag_dx = half_dx * np.sqrt(2)
        cell_area = self[-1].dx[0] ** 2

        intersect_cell_idx = (dist2origin > radius - half_diag_dx) & (dist2origin < radius + half_diag_dx)

        circle = g.Point(*tuple(origin)).buffer(radius, resolution=shapely_res)
        cell_polygons = [g.Polygon(self.cell_vertices(xy, half_dx)) for xy in self.finest_meshXY[intersect_cell_idx, :]]
        intersect_area = np.array([circle.intersection(c).area for c in cell_polygons])

        self.polar_fraction = np.zeros_like(self._data_dict[self.finest_names[0]], dtype=float)
        self.non_polar_fraction = np.zeros_like(self._data_dict[self.finest_names[0]], dtype=float)
        self.polar_fraction[dist2origin < radius - half_diag_dx] = 1
        self.polar_fraction[intersect_cell_idx] = intersect_area / cell_area
        self.non_polar_fraction = 1 - self.polar_fraction

        return self.polar_fraction, self.non_polar_fraction


class AthenaLIS:
    """ Read data from LIS files from SI simulations (by Athena)
        AthenaLIS is able to read BINARY particle data
        e.g.,
            >>> a = AthenaLIS("Par_Start3d.0000.ds.lis")
                Read 2097152 particles.
            >>> a[123]['vel']
                array([0.00281672, 0.03813027, 0.00299831], dtype=float32)
            >>> a[:]['pos'].shape
                (1048576, 3)
        :param filename: the name of your data file
        :param silent: default True; if False, will output reading info
        :param memmapping: default True, will use np.memmap for large data file (>1e7 particles)
        :param sort: default False; if True, uni_ids = particle idx sorted by cpu_id and then by id
        :param fp24: default False; if True, the LIS file contains float24 instead of float
    """

    ori_dtype = np.dtype([('pos', 'f4', 3),  # original dtype in LIS files output by athena
                          ('vel', 'f4', 3),
                          ('den', 'f4'),
                          ('property_index', 'i4'),
                          ('id', 'i8'),
                          ('cpu_id', 'i4')])
    fp24_dtype = np.dtype([('pos', 'u1', 9),  # float24 = 3 unsigned ints
                           ('vel', 'u1', 9),
                           ('den', 'u1', 3),
                           ('property_index', 'i4'),
                           ('id', 'i8'),
                           ('cpu_id', 'i4')])

    def __init__(self, filename, silent=True, memmapping=True, sort=False, custom_dtype=None):
        """ directly read binary particle data """

        f = open(filename, 'rb')
        if custom_dtype is not None:
            self.dtype = custom_dtype
        else:
            try:
                self.dtype = read_dtype_from_file(f)
            except ValueError:
                self.dtype = self.ori_dtype
                f.seek(0, 0)
            except Exception as e:
                raise RuntimeError("Attempt read_dtype_from_file() failed.") from e

        fp24 = False
        if self.dtype.descr[:3] == [('pos', '|u1', (9,)), ('vel', '|u1', (9,)), ('den', '|u1', (3,))]:
            fp24 = True
        if fp24 is False and self.dtype['pos'].base == np.dtype('uint8'):
            raise TypeError("It seems fp24=False but dtype['pos'] has uint8? dtype=", self.dtype)

        self.coor_lim = np.array(readbin(f, '12f'))
        self.box_min = np.array(self.coor_lim[6:11:2])
        self.box_max = np.array(self.coor_lim[7:12:2])
        self.dim = np.count_nonzero(self.box_max - self.box_min)

        self.num_types = readbin(f, 'i')
        self.type_info = np.array(readbin(f, str(self.num_types)+'f'))
        self.t, self.dt = readbin(f, '2f')
        self.num_particles = readbin(f, 'l')

        if memmapping or self.num_particles > 1e7:
            offset_needed = f.tell()
            self.particles = np.memmap(filename, mode='r', offset=offset_needed, dtype=self.dtype)
        else:
            self.particles = np.fromfile(f, dtype=self.dtype)

        # float24 (3 uint8) needs to be converted to float32 for native data analysis
        if fp24:
            if 'property_index' in self.dtype.names:
                int_dtype = np.dtype([(name, self.dtype[name].str) for name in ['property_index', 'id', 'cpu_id']])
            else:
                int_dtype = np.dtype([(name, self.dtype[name].str) for name in ['id', 'cpu_id']])
            fp_dtype = np.dtype(self.ori_dtype.descr[:3])

            new_particles = np.zeros(self.num_particles, dtype=np.dtype(fp_dtype.descr + int_dtype.descr))
            new_particles['pos'] = convert_array_float24_to_float32(
                self.particles['pos'].flatten().reshape([-1, 3])).reshape([-1, 3])
            new_particles['vel'] = convert_array_float24_to_float32(
                self.particles['vel'].flatten().reshape([-1, 3])).reshape([-1, 3])
            new_particles['den'] = convert_array_float24_to_float32(
                self.particles['den'].flatten().reshape([-1, 3]))
            if 'property_index' in self.dtype.names:
                new_particles['property_index'] = self.particles['property_index']
            new_particles['id'] = self.particles['id']
            new_particles['cpu_id'] = self.particles['cpu_id']
            self.dtype = new_particles.dtype
            del self.particles
            self.particles = new_particles

        if sort:
            self.uni_ids = np.argsort(self.particles, order=['cpu_id', 'id'])

        f.close()
        if not silent:
            print("Read "+str(self.num_particles)+" particles.")

    def __getitem__(self, index):
        """ Overload indexing operator [] for particles """

        return self.particles[index]

    def sub_sampling(self, filename, rate):
        """ Sub-sampling to a smaller data file """

        f = open(filename, 'wb')

        f.write(self.coor_lim.astype('f').tobytes())
        writebin(f, self.num_types, 'i')
        f.write(self.type_info.astype('f').tobytes())
        writebin(f, self.t, 'f')
        writebin(f, self.dt, 'f')

        assert (rate >= 1)
        sub_sample = self.particles[self.particles['id'] < rate]
        writebin(f, sub_sample.size, 'l')
        print("Sub-sampling "+str(sub_sample.size)+" particles to "+filename)
        sub_sample.tofile(f)

        f.close()

    def restore_trimmed_LIS(self, filename):
        """ Restore the trimmed LIS data set back to ori_dtype """

        out_particles = np.zeros(self.num_particles, dtype=self.ori_dtype)
        out_particles['pos'] = self.particles['pos']
        out_particles['vel'] = self.particles['vel']
        out_particles['den'] = self.particles['den']
        if 'property_index' in self.dtype.names:
            out_particles['property_index'] = self.particles['property_index']
        # no need to else as they are zeros by default
        out_particles['id'] = self.particles['id']
        out_particles['cpu_id'] = self.particles['cpu_id']

        f = open(filename, 'wb')
        f.write(self.coor_lim.astype('f').tobytes())
        writebin(f, self.num_types, 'i')
        f.write(self.type_info.astype('f').tobytes())
        writebin(f, self.t, 'f')
        writebin(f, self.dt, 'f')
        writebin(f, self.num_particles, 'l')
        out_particles.tofile(f)
        f.close()
        print(f"Trimmed LIS data restored to {filename}")

    def to_point3d_file(self, filename, sampling=None):
        """ Write particle data to a POINT3D file for visualization """

        pos = self.particles['pos']
        p_id = self.particles['id']
        if sampling is None:
            dump = np.append(pos, np.atleast_2d(p_id).T, axis=1)
        else:
            if isinstance(sampling, int):
                assert(sampling > 1)
                dump = np.append(pos[::sampling], np.atleast_2d(p_id[::sampling]).T, axis=1)
            else:
                raise TypeError("sampling should be a positive integer, sampling = ", sampling)
        np.savetxt(filename, dump, fmt = "%15e"*3+"  %08d", header="# POINT3D file from t={:.3f}".format(self.t))

    def make_ghost_particles(self, q, time):
        """ Make ghost particles based on the shear parameter q and time """

        raise NotImplementedError("Making ghost particles has not been implemented.")

    def plot_scatter(self, normal='z', sampling_rate=1, s=0.025, c='b', ax=None, figsize=None):
        """
        Plot a scatter figure of particles
        :param normal: the normal direction of the projection (e.g., 'z', 'y')
        :param sampling_rate: plot every one of X particles
        :param s: marker size for scatter plot
        :param ax: Axes object for plotting; will create one if None
        :param figsize: customized figure size
        """

        new_ax_flag = False
        if ax is None:
            plt_params("medium")
            fig, ax = plt.subplots(figsize=figsize)
            new_ax_flag = True

        if normal == 'z':
            ax.scatter(self['pos'][::sampling_rate, 0], self['pos'][::sampling_rate, 1], s=s,
                       marker='o', facecolors=c, edgecolors='None')
            if new_ax_flag: ax_labeling(ax, x=r"$x/H$", y=r"$y/H$")
        elif normal == 'y':
            ax.scatter(self['pos'][::sampling_rate, 0], self['pos'][::sampling_rate, 2], s=s,
                       marker='o', facecolors=c, edgecolors='None')
            if new_ax_flag: ax_labeling(ax, x=r"$x/H$", y=r"$z/H$")
        elif normal == 'x':
            ax.scatter(self['pos'][::sampling_rate, 1], self['pos'][::sampling_rate, 2], s=s,
                       marker='o', facecolors=c, edgecolors='None')
            if new_ax_flag: ax_labeling(ax, x=r"$y/H$", y=r"$z/H$")
        else:
            raise ValueError("keyword normal can only be 'z', 'y', or 'x'.")

        if new_ax_flag:
            ax.set_aspect(1.0)
            return fig, ax


def trim_LIS(filename, out_filename, cut_fp="None", cut_int=False, **kwargs):
    """ Trim LIS file size by converting fp32 to fp24
        :param filename: the file name of the original LIS file to read
        :param out_filename: the output file for trimmed LIS data
    """
    if cut_fp == "None" and cut_int is False:
        print("cut_fp='None', cut_int=False, nothing to trim.")
        return None
    if not isinstance(cut_fp, str):
        raise ValueError("cut_fp must be a string, options are ['fp24', 'fp16', 'float24', 'float16']")

    read_kwargs = kwargs.get('read_kw', {})
    ds = AthenaLIS(filename, **read_kwargs)  # ds = dataset

    skip_cut_fp = False
    if cut_fp == "fp24" or cut_fp == "float24":
        cut_pos = convert_array_float32_to_float24(ds['pos'].flatten()).reshape([-1, 9])
        cut_vel = convert_array_float32_to_float24(ds['vel'].flatten()).reshape([-1, 9])
        cut_den = convert_array_float32_to_float24(ds['den'].flatten()).reshape([-1, 3])
        fp_dtype = np.dtype([('pos', 'u1', 9), ('vel', 'u1', 9), ('den', 'u1', 3)])
    elif cut_fp == "fp16" or cut_fp == "float16":
        cut_pos = ds['pos'].astype(np.float16)
        cut_vel = ds['pos'].astype(np.float16)
        cut_den = ds['den'].astype(np.float16)
        fp_dtype = np.dtype([('pos', 'f2', 3), ('vel', 'f2', 3), ('den', 'f2')])
    else:
        fp_dtype = np.dtype([('pos', 'f4', 3), ('vel', 'f4', 3), ('den', 'f4')])
        skip_cut_fp = True

    if cut_int:
        # Choose the smallest data type that can accommodate the range of values for each field
        id_dtype = get_minimum_signed_dtype(ds['id'].min(), ds['id'].max())
        cpu_id_dtype = get_minimum_signed_dtype(ds['cpu_id'].min(), ds['cpu_id'].max())
        if ds.num_types > 1:
            property_index_dtype = get_minimum_signed_dtype(ds['property_index'].min(), ds['property_index'].max())
            int_dtype = np.dtype([('property_index', property_index_dtype), ('id', id_dtype), ('cpu_id', cpu_id_dtype)])
        else:
            int_dtype = np.dtype([('id', id_dtype), ('cpu_id', cpu_id_dtype)])
    else:
        int_dtype = np.dtype([('property_index', 'i4'), ('id', 'i8'), ('cpu_id', 'i4')])

    all_dtype = np.dtype(fp_dtype.descr + int_dtype.descr)  # one can check if 'property_index' in all_dtype.names
    # Create a new structured array to store the converted data
    new_particles = np.zeros(ds.num_particles, dtype=all_dtype)
    if skip_cut_fp:
        new_particles['pos'] = ds['pos']
        new_particles['vel'] = ds['vel']
        new_particles['den'] = ds['den']
    else:
        new_particles['pos'] = cut_pos
        new_particles['vel'] = cut_vel
        new_particles['den'] = cut_den
    if cut_int:
        if ds.num_types > 1:
            new_particles['property_index'] = ds.particles['property_index'].astype(property_index_dtype)
        new_particles['id'] = ds.particles['id'].astype(id_dtype)
        new_particles['cpu_id'] = ds.particles['cpu_id'].astype(cpu_id_dtype)
    else:
        if 'property_index' in ds.dtype.names:
            new_particles['property_index'] = ds['property_index']
        new_particles['id'] = ds['id']
        new_particles['cpu_id'] = ds['cpu_id']

    # Now save the new data to a new LIS file
    with open(out_filename, 'wb') as f:
        write_dtype_to_file(all_dtype, f)
        f.write(ds.coor_lim.astype('f').tobytes())
        writebin(f, ds.num_types, 'i')
        f.write(ds.type_info.astype('f').tobytes())
        writebin(f, ds.t, 'f')
        writebin(f, ds.dt, 'f')

        writebin(f, ds.num_particles, 'l')
        new_particles.tofile(f)
        # RL: looks like some routes may produce different binary files but the content data are identical
        # other methods of dumping, e.g., f.write(new_particles.tobytes()) or np.ascontiguousarray do not help
    print(f"Converted data (with cut_fp={cut_fp}, cut_int={cut_int}) saved to {out_filename}")


class AthenaMultiLIS(AthenaLIS):
    """ Read data from sub-LIS files from all processors from SI simulations (by Athena)
            AthenaMultiLIS is able to read BINARY particle data (it is always output as 3D)
            e.g.,
                >>> a = AthenaMultiLIS("bin", "Par_Strat3d", "0001.ds.lis", silent=False)
                    Read 2097152 particles.
    """

    def __init__(self, data_dir, prefix, postfix, silent=True, sort=False):

        id_folders = [x for x in os.listdir(data_dir) if x[:2] == 'id']
        if len(id_folders) == 0:
            raise RuntimeError("No data files to read (no id*)")

        self.num_cpus = len(id_folders)

        if data_dir[-1] != '/':
            data_dir = data_dir + '/'

        filenames = [data_dir + "id" + str(x) + '/' + prefix + "-id" + str(x) + '.' + postfix for x in
                     range(self.num_cpus)]
        filenames[0] = data_dir + "id0/" + prefix + '.' + postfix

        tmp_data = AthenaLIS(filenames[0])
        self.box_min = np.array(tmp_data.coor_lim[6:11:2])
        self.box_max = np.array(tmp_data.coor_lim[7:12:2])
        self.dim = np.count_nonzero(self.box_max - self.box_min)
        self.coor_lim = np.hstack([tmp_data.coor_lim[6:], tmp_data.coor_lim[6:]])
        self.t = tmp_data.t
        self.dt = tmp_data.dt
        self.num_types = tmp_data.num_types
        self.type_info = tmp_data.type_info

        self.particles = np.hstack([AthenaLIS(idx, memmapping=False).particles for idx in filenames])
        self.num_particles = self.particles.size
        if sort:
            self.uni_ids = np.argsort(self.particles, order=['cpu_id', 'id'])

        if not silent:
            print("Read " + str(self.num_particles) + " particles.")

    def __getitem__(self, index):
        """ Overload indexing operator [] for particles """

        return self.particles[index]
