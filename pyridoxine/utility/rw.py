""" Provide convenient reading/writing functions """

import traceback
import struct
from array import array
from numbers import Number
import subprocess as subp
import numpy as np
import pandas as pd
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
    """ Load a sequence of data from a binary file, return an ndarray """

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

        tmp_line = f.readline().decode('utf-8')
        assert(tmp_line[:9] == "CELL_DATA"), "no CELL_DATA info: "+tmp_line
        self.size = int(tmp_line[10:])

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
            "dpar": ["particle_density"],
            "particle_density": ["dpar"],
            "rhop": ["dpar", "particle_density"],
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
        self.__simplified_components = ["ux", "uy", "uz",
                                        "vx", "vy", "vz",
                                        "wx", "wy", "wz",
                                        "Bx", "By", "Bz"]
        self.__common_types = {'float': 'f', 'double': 'd', 'int': 'i'}
        real_wanted = []
        if wanted is not None:
            for i, item in enumerate(wanted):
                if item in self.__simplified_names:
                    real_wanted += self.__simplified_names[item]
                else:
                    real_wanted += [item]

        while f.tell() != eof:
            tmp_line = f.readline().decode('utf-8')
            if tmp_line == '\n':
                tmp_line = f.readline().decode('utf-8')
            tmp_line = tmp_line.split()
            if wanted is not None:
                if tmp_line[1] not in real_wanted:
                    if tmp_line[0] == "SCALARS":
                        f.readline()  # skip "LOOKUP_TABLE default"
                        if tmp_line[2] in self.__common_types:
                            f.seek(self.size * struct.calcsize(self.__common_types[tmp_line[2]]), 1)
                        else:
                            f.seek(self.size * struct.calcsize('f'), 1)
                    if tmp_line[0] == "VECTORS":
                        if tmp_line[2] in self.__common_types:
                            f.seek(3 * self.size * struct.calcsize(self.__common_types[tmp_line[2]]), 1)
                        else:
                            f.seek(3 * self.size * struct.calcsize('f'), 1)
                    continue

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
                else:
                    tmp_data = array('f')
                tmp_data.fromfile(f, self.size)
                self.data[tmp_line[1]] = np.asarray(tmp_data).byteswap().reshape(np.flipud(self.Nx[:self.dim]))

            elif tmp_line[0] == "VECTORS":
                if tmp_line[2] == "float":
                    tmp_data = array('f')
                elif tmp_line[2] == "double":
                    tmp_data = array('d')
                elif tmp_line[2] == "int":
                    tmp_data = array('i')
                else:
                    tmp_data = array('f')
                tmp_data.fromfile(f, self.size*3)  # even for 2D simulations, the vector fields are 3D
                tmp_shape = np.hstack([np.flipud(self.Nx[:self.dim]), 3])
                self.data[tmp_line[1]] = np.asarray(tmp_data).byteswap().reshape(tmp_shape)

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

            raise KeyError(data_name+" not found. Available are "+", ".join(self.names))

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
            warnings.warn("It seems previous ghost zones used a different ghost width: ", self.ghost_width,
                          ".\nUsing a new width will update the cell-center coordinates ccx/y/z_gh and l/r_corner_gh.")

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
            if log_norm: data = np.log10(data / data.mean())
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
                ax.pcolorfast(self.ccx, self.ccz, data, **kwargs)
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
                ax.pcolorfast(self.ccy, self.ccz, data, **kwargs)
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

class AthenaMultiVTK(AthenaVTK):
    """ Read data from sub-VTK files from all processors from SI simulations (by Athena)
        AthenaMultiVTK is able to read BINARY data of STRUCTURED_POINTS, either 2D or 3D
        e.g.,
            >>> a = AthenaMultiVTK("bin", "Par_Strat3d", "0001.vtk", silent=False)
            Read [density, momentum, particle_density, particle_momentum] at Nx=[64, 64, 64]

    """

    def __init__(self, data_dir, prefix, postfix, wanted=None, xyz_order=None, silent=True):

        id_folders = [x for x in os.listdir(data_dir) if x[:2] == 'id']
        if len(id_folders) == 0:
            raise RuntimeError("No data files to read (no id*)")

        self.num_cpus = len(id_folders)

        if data_dir[-1] != '/':
            data_dir = data_dir + '/'

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
    """

    def __init__(self, filename, silent=True, memmapping=True, sort=False):
        """ directly read binary particle data """

        f = open(filename, 'rb')

        self.coor_lim = np.array(readbin(f, '12f'))
        self.box_min = np.array(self.coor_lim[6:11:2])
        self.box_max = np.array(self.coor_lim[7:12:2])

        self.num_types = readbin(f, 'i')
        self.type_info = np.array(readbin(f, str(self.num_types)+'f'))
        self.t, self.dt = readbin(f, '2f')
        self.num_particles = readbin(f, 'l')

        self.dtype = np.dtype([('pos', 'f4', 3),
                               ('vel', 'f4', 3),
                               ('den', 'f4'),
                               ('property_index', 'i4'),
                               ('id', 'i8'),
                               ('cpu_id', 'i4')])

        if memmapping or self.num_particles > 1e7:
            offset_needed = f.tell()
            self.particles = np.memmap(filename, mode='r', offset=offset_needed, dtype=self.dtype)
        else:
            self.particles = np.fromfile(f, dtype=self.dtype)
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
