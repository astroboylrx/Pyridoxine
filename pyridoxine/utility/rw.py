""" Provide convenient reading/writing functions """

__valid_array_typecode = ['b', 'B', 'u', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'f', 'd']

import traceback
import struct
from array import array
from numbers import Number
import subprocess as sp
import numpy as np
import pandas as pd
import yt

def __read_formatted_column_super_slow(filepath, col2read):
    """ Read one data column from a formatted text file
        This is still way more slower than loadtxt.
        Instead of building wheels, I found genfromtxt is a little bit faster.
        Then I found pandas.read_csv(...).as_matrix() is way more faster.
    """

    line_length = 0
    num_lines = int(sp.getoutput("wc -l "+filepath).split()[0])
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
                if (col2read >= len(a)):
                    raise ValueError("There is no such column to read: ", col2read)
                break

    if line_length is 0:
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
        if first_data_line[idx] is not ' ':
            while idx < len(first_data_line) and first_data_line[idx] is not ' ':
                idx += 1
            goods += 1
            continue
        else:
            if goods == col2read:
                col_start = idx
            if goods == col2read+1:
                col_end = idx
                break
            while idx < len(first_data_line) and first_data_line[idx] is ' ':
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


def loadtxt(filepath, h=0, c=None):
    """ Wrapping pandas.read_csv()
        h: how many rows to skip in order to reach real data
        c: which column(s) to read
    """

    if c is None:
        return pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=h).as_matrix()
    else:
        if isinstance(c, Number):
            c = [c]
        return pd.read_csv(filepath, delim_whitespace=True, header=None, skiprows=h, usecols=c).as_matrix()


def readbin(file_handler, dtype='d', size=8):
    """ Retrieve a certain type of data from a binary file, just one item """

    if not isinstance(dtype, str):
        raise TypeError("reading format need to be a str: ", dtype)
    if len(dtype) != 1:
        raise ValueError("bad char in reading format: ", dtype)
    if not isinstance(size, Number):
        raise TypeError("size should be a scalar, but got: ", size)
    if size < 0:
        raise ValueError("size should be larger than 0, but got: ", size)

    ori_pos = file_handler.tell()
    try:
        data = struct.unpack(dtype, file_handler.read(size))[0]
        return data
    except Exception:
        traceback.print_exc()
        print("Rolling back to the original stream position...")
        file_handler.seek(ori_pos)
        return None


def loadbin(file_handler, dtype='d', num=1):
    """ Load a sequence of data from a binary file, return an ndarray """

    if not isinstance(dtype, str):
        raise ValueError("typecode need to be a str: ", dtype)
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


class __YTLoadAthenaVTK:
    """ Use yt to read data from vtk file in SI simulations (by Athena)
        YT follows the name convention of yt.YTArray
        This turns out to be slow and mysteriously a memory hog sometimes.
    """

    def __init__(self, filename):
        """ load data from VTK file using yt """

        from yt.funcs import mylog
        mylog.setLevel(40)  # This sets the log level to "ERROR"

        self.pf = yt.load(filename)  # load() always gives memory increment due to sympy units system
        data = self.pf.index.grids[0]
        self.t = self.pf.current_time
        self.Nx = self.pf.domain_dimensions

        if self.pf.field_list == [('athena','dpar')]:
            self.rhop = np.squeeze(data['dpar'])
        elif self.pf.field_list == [('athena', 'density'),
                               ('athena', 'momentum_x'),
                               ('athena', 'momentum_y'),
                               ('athena', 'momentum_z'),
                               ('athena', 'particle_density'),
                               ('athena', 'particle_momentum_x'),
                               ('athena', 'particle_momentum_y'),
                               ('athena', 'particle_momentum_z')]:
            self.rhog, self.ux, self.uy, self.uz, self.rhop, self.wx, self.wy, self.wz = \
                [np.squeeze(data[field[1]]) for field in self.pf.field_list]
            # RL: np.squeeze(a) always returns a itself or a view into a
            #     However, this is still a super memory hog!
            #     A snapshot of 256^3 run need at least 1.41 GB memory to store (VTK only 512M)!
        else:
            print("Retrieved field:", self.pf.field_list)
            self.data = data


class AthenaVTK:
    """ Read data from VTK files from SI simulations (by Athena)
        AthenaVTK is able to read BINARY data of STRUCTURED_POINTS, either 2D or 3D
        e.g.,
            >>> a = AthenaVTK("Cout.0500.vtk", silent=False)
            Read [density, momentum, particle_density, particle_momentum] at Nx=[64, 64, 64]
    """

    def __init__(self, filename, silent=True):
        """ directly read binary data """

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

        tmp_line = f.readline().decode('utf-8')  # normally "DIMENSIONS 129 65 1"
        self.Nx = [int(x) - 1 for x in tmp_line[11:].split()]
        if self.Nx[2] == 0:
            self.dim = 2
        else:
            self.dim = 3

        tmp_line = f.readline().decode('utf-8')
        assert (tmp_line[:6] == "ORIGIN"), "no ORIGIN info: "+tmp_line
        self.left_corner = [float(x) for x in tmp_line[7:].split()]

        tmp_line = f.readline().decode('utf-8')
        assert (tmp_line[:7] == "SPACING"), "no SPACING info: "+tmp_line
        self.dx = [float(x) for x in tmp_line[8:].split()]

        tmp_line = f.readline().decode('utf-8')
        assert(tmp_line[:9] == "CELL_DATA"), "no CELL_DATA info: "+tmp_line
        self.size = int(tmp_line[10:])

        self.svtypes = []
        self.names = []
        self.dtypes = []
        self.data = dict()

        while f.tell() != eof:
            tmp_line = f.readline().decode('utf-8')
            if tmp_line == '\n':
                tmp_line = f.readline().decode('utf-8')
            tmp_line = tmp_line.split()
            self.svtypes.append(tmp_line[0])
            self.names.append(tmp_line[1])
            self.dtypes.append(tmp_line[2])
            if tmp_line[0] == "SCALARS":
                f.readline()  # skip "LOOKUP_TABLE default"
                if tmp_line[2] == "float":
                    tmp_data = array('f')
                elif tmp_line[2] == "double":
                    tmp_data = array('d')
                elif tmp_line[2] == "int":
                    tmp_data = array('i')
                tmp_data.fromfile(f, self.size)
                self.data[tmp_line[1]] = np.asarray(tmp_data).byteswap().reshape(np.flipud(self.Nx[:self.dim]))

            elif tmp_line[0] == "VECTORS":
                if tmp_line[2] == "float":
                    tmp_data = array('f')
                elif tmp_line[2] == "double":
                    tmp_data = array('d')
                elif tmp_line[2] == "int":
                    tmp_data = array('i')
                tmp_data.fromfile(f, self.size*3)
                tmp_shape = np.hstack([np.flipud(self.Nx[:self.dim]), 3])
                self.data[tmp_line[1]] = np.asarray(tmp_data).byteswap().reshape(tmp_shape)

        f.close()
        if not silent:
            print("Read ["+", ".join(self.names)+"] at Nx=["+", ".join([str(x) for x in self.Nx])+"]")

    def __getitem__(self, data_name):
        """ Overload indexing operator [] """

        if data_name in self.data:
            return self.data[data_name].view()
        else:
            raise KeyError(data_name+" not found. Available are "+", ".join(self.names))

    def __setitem__(self, data_name, value):
        """ Overload writing access by operator [] """

        raise IOError("Writing access to data is not implemented.")
