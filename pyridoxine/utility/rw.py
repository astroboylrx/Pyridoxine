""" Provide convenient reading/writing functions """

import numpy as np
import subprocess as sp
import pandas as pd
from numbers import Number
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


class AthenaVtkData:
    """ Use yt to read data from vtk file in SI simulations """

    def __init__(self, filename):
        """ load data from VTK file using yt """

        from yt.funcs import mylog
        mylog.setLevel(40)  # This sets the log level to "ERROR"

        self.pf = yt.load(filename)
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
            # RL: np.squeeze always a itself or a view into a.
        else:
            print("Retrieved field:", self.pf.field_list)
            self.data = data



