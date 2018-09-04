""" Provide vector classes and related calculations for arrays """

import numpy as np
import copy
from numbers import Number
import warnings
import operator
from .constants import c


class Vector:
    """ Define a vector class for vector calculation. """

    def __init__(self, *args, dtype=None):
        """
        Construct a vector (currently only support 2D and 3D)
        :param args: x1, x2[, x3] or numpy.ndarray or list or tuple
        """

        if len(args) is 0:
            raise SyntaxError("Vector must take [an] argument[s] to proceed.")

        if dtype is None:
            self.dtype = float
        else:
            self.dtype = dtype

        self.data = np.array([], dtype=self.dtype)

        for item in self.traverse(args):
            if isinstance(item, Number):
                self.data = np.append(self.data, self.dtype(item))
            else:
                raise TypeError("Cannot establish a vector with non-numeric argument: ", item)

        """ original method
        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                self.data = np.append(self.data, (args[i]).flatten())
            elif isinstance(args[i], (list, tuple)):
                self.data = np.append(self.data, np.asarray(args[i]).flatten())
            elif isinstance(args[i], Number):
                self.data = np.append(self.data, args[i])
            else:
                raise TypeError("Cannot establish a vector with non-numeric argument: ", args[i])
        """

        self.dim = self.data.size
        self.__iter_idx = 0  # index used to make Vector iterable

        if self.dim is 0:  # should not happen
            raise ValueError("Cannot establish a vector without numbers")
        if self.dim is 1:
            raise ValueError("Cannot establish a vector with one scalar.")
        elif self.dim > 3:
            self.data = self.data[:3]
            self.dim = 3
            warnings.warn("""
            Only the first three numbers in arguments are effective. 
            Vector only support 2D or 3D vectors.
            """, RuntimeWarning)

        self.r = np.sqrt(np.sum(self.data**2))
        self.phi = np.arctan2(self.data[1], self.data[0])
        if self.dim is 3:
            self.theta = np.arccos(self.data[2]/self.r)

    def traverse(self, food):
        """ traverse arbitrarily nested list/tuple/array
            ref: https://stackoverflow.com/a/6340578/4009531
        """

        if isinstance(food, (np.ndarray, list, tuple, Vector)):
            for value in food:
                for subvalue in self.traverse(value):
                    yield subvalue
        else:
            yield food

    def traverse_vector(self, food):
        """ traverse arbitrarily nested list/tuple/array
            ref: https://stackoverflow.com/a/6340578/4009531
        """

        if isinstance(food, (np.ndarray, list, tuple)):
            for value in food:
                for subvalue in self.traverse_vector(value):
                    yield subvalue
        elif isinstance(food, Vector):
            yield food
        else:
            raise TypeError("Not a Vector: ", food)

    def calculate_r_angles(self):
        """ Compute |r| and angles once data has changed. """

        self.r = np.sqrt(np.sum(self.data ** 2))
        self.phi = np.arctan2(self.data[1], self.data[0])
        if self.dim is 3:
            self.theta = np.arccos(self.data[2] / self.r)

    def __arithmetic(self, other, operation, op_name=None):
        """ Perform arithmetic operation """

        temp = copy.deepcopy(self)
        if isinstance(other, Vector):
            if other.dim != self.dim:
                raise ValueError("Dimension mismatch: ", self.dim, other.dim)
            temp.data = operation(temp.data, other.data)
        elif isinstance(other, (np.ndarray, list, tuple)):
            try:
                # use "and" in determination instead of "&" to get the correct result
                if all((isinstance(item, Vector) and item.dim == self.dim) for item in self.traverse_vector(other)):
                    temp = copy.deepcopy(other)
                    # this for loop indeed changes element since it is immutable
                    for element in self.traverse_vector(temp):
                        element.data = operation(element.data, self.data)
                        element.calculate_r_angles()
                    return temp
            except TypeError:
                # np.asarray(Vector) = Vector.data; but mixed types may confuse asarray
                other = np.asarray([item for item in self.traverse(other)]).flatten()
                if other.size != self.dim:
                    raise ValueError("Dimension mismatch: ", self.dim, other.size)
                temp.data = operation(temp.data, other)
        elif isinstance(other, Number):
            temp.data = operation(temp.data, other)
        else:
            if op_name is None:
                raise TypeError("Cannot perform ", operation, " on ", self, " and ", other)
            else:
                raise TypeError("Cannot perform ", op_name, " on ", self, " and ", other)

        temp.calculate_r_angles()
        return temp

    def __abs__(self):
        """ Overload absolute |self| = r """

        return self.r

    def __pos__(self):
        """ Overload absolute +self """

        return self

    def __pow__(self, power):
        """ Overload power self**p = r**p """

        if not isinstance(power, Number):
            raise TypeError("Cannot calculate a Vector to the power of ", power)
        return self.r**power

    def __add__(self, other):
        """ Overload the return value of self + other """

        return self.__arithmetic(other, operator.add)

    def __radd__(self, other):
        """ Overload the return value of other + self """

        return self.__arithmetic(other, operator.add)

    def __iadd__(self, other):
        """ Overload self += other. """

        return self.__arithmetic(other, operator.add)

    def __mul__(self, other):
        """ Overload the return value of parallel product: self * other """

        return self.__arithmetic(other, operator.mul)

    def __rmul__(self, other):
        """ Overload the return value of parallel product: other * self """

        return self.__arithmetic(other, operator.mul)

    def __imul__(self, other):
        """ Overload self *= other """

        return self.__arithmetic(other, operator.mul)

    def __sub__(self, other):
        """ Overload the return value of self - other """

        return self.__arithmetic(other, operator.sub)

    def __rsub__(self, other):
        """ Overload the return value of other - self """

        temp = copy.deepcopy(self)
        temp.data *= (-1)
        return temp.__arithmetic(other, operator.add, op_name=r"subtraction")

    def __isub__(self, other):
        """ Overload self -= other """

        return self.__arithmetic(other, operator.sub)

    def __truediv__(self, other):
        """ Overload division: self / other """

        if isinstance(other, Vector):
            if self.dim is 2 and other.phi != self.phi:
                raise ValueError("Vectors in division must have the same angle[s].")
            if self.dim is 3 and (other.phi != self.phi or other.theta != self.theta):
                raise ValueError("Vectors in division must have the same angle[s].")
            return self.r / other.r
        elif isinstance(other, Number):
            return self.__mul__(1. / other)
        elif isinstance(other, (np.ndarray, list, tuple)):
            warnings.warn("Dividing Vector by array/list/tuple (element by element)")
            other = np.asarray([item for item in self.traverse(other)]).flatten()
            if other.size != self.dim:
                raise ValueError("Dimension mismatch: ", self.dim, other.size)
            return self.__mul__(1. / other)
        else:
            raise TypeError("Cannot divide ", other, " from Vector ", self.data)

    def __rtruediv__(self, other):
        """ Overload division: other / self """

        if isinstance(other, Vector):
            if self.dim is 2 and other.phi != self.phi:
                raise ValueError("Vectors in division must have the same angle[s].")
            if self.dim is 3 and (other.phi != self.phi or other.theta != self.theta):
                raise ValueError("Vectors in division must have the same angle[s].")
            return other.r / self.r
        elif isinstance(other, Number):
            raise TypeError("Dividing a Number by a Vector is not defined.")
        elif isinstance(other, (np.ndarray, list, tuple)):
            warnings.warn("""Dividing array/list/tuple by Vector (element by element)""")
            other = np.asarray([item for item in self.traverse(other)]).flatten()
            if other.size != self.dim:
                raise ValueError("Dimension mismatch: ", self.dim, other.size)
            return Vector(other / self.data)
        else:
            raise TypeError("Cannot divide ", other, " by Vector ", self.data)

    def __itruediv__(self, other_value):
        """ Overload self /= other_value """

        return self.__truediv__(other_value)

    def __lt__(self, other):
        """ Overload comparison operator < """

        if not isinstance(other, Vector):
            raise TypeError("Comparison can only be done between Vectors")
        if self.dim != other.dim:
            raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
        return self.r < other.r

    def __le__(self, other):
        """ Overload comparison operator < """

        if not isinstance(other, Vector):
            raise TypeError("Comparison can only be done between Vectors")
        if self.dim != other.dim:
            raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
        return self.r <= other.r

    def __eq__(self, other):
        """ Overload comparison operator < """

        if not isinstance(other, Vector):
            raise TypeError("Comparison can only be done between Vectors")
        if self.dim != other.dim:
            raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
        return self.r == other.r

    def __ne__(self, other):
        """ Overload comparison operator < """

        if not isinstance(other, Vector):
            raise TypeError("Comparison can only be done between Vectors")
        if self.dim != other.dim:
            raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
        return self.r != other.r

    def __gt__(self, other):
        """ Overload comparison operator < """

        if not isinstance(other, Vector):
            raise TypeError("Comparison can only be done between Vectors")
        if self.dim != other.dim:
            raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
        return self.r > other.r

    def __ge__(self, other):
        """ Overload comparison operator < """

        if not isinstance(other, Vector):
            raise TypeError("Comparison can only be done between Vectors")
        if self.dim != other.dim:
            raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
        return self.r >= other.r

    def __str__(self):
        """ Called by str(object) and the built-in functions format() and print() """

        return str(self.data)

    def __repr__(self):
        """ Change the official string representation from object name to data """

        return str(self.data)

    def __getitem__(self, index):
        """ Overload indexing operator [] """

        if index >= self.dim:
            raise ValueError("Bad outbound access (overflow); ", index, ">=", self.dim)
        return self.data[index]

    def __setitem__(self, index, value):
        """ Overload writing access by operator [] """

        if index >= self.dim:
            raise ValueError("Bad outbound access (overflow); ", index, ">=", self.dim)
        self.data[index] = value

    def __iter__(self):
        return self

    def __next__(self):
        """ Make Vector iterable """

        self.__iter_idx += 1
        try:
            return self.data[self.__iter_idx - 1]
        except IndexError:
            self.idx = 0
            raise StopIteration  # Done iterating

    def show(self):
        """ Print vector info """

        if self.dim is 2:
            print(r"(x, y)=({0:.6e}, {1:.6e}), |r|={2:.6e}, phi={3:.6e}[rad]/{4:.6e}[deg]".format(
            self.data[0], self.data[1], self.r, self.phi, self.phi*180/np.pi))
        elif self.dim is 3:
            print(r"(x, y, z)=({0:.6e}, {1:.6e}, {2:.6e}), |r|={3:.6e}, \
                   theta={4:.6e}[rad]/{5:.6e}[deg], phi={6:.6e}[rad]/{7:.6e}[deg]".format(
                self.data[0], self.data[1], self.data[2], self.r,
                self.theta, self.theta * 180 / np.pi, self.phi, self.phi * 180 / np.pi))

    def is_parallel(self, other):
        """ Determine if parallel to another Vector"""

        if not isinstance(other, Vector):
            raise TypeError("Comparison can only be done between Vectors")
        if self.dim != other.dim:
            raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
        if self.dim is 2:
            return self.phi == other.phi
        if self.dim is 3:
            return self.phi == other.phi and self.theta == other.theta

    def dot(self, other):
        """ define dot product of two Vectors """

        temp = self.__arithmetic(other, operator.mul, op_name=r"dot product")
        if isinstance(temp, Vector):
            return np.sum(temp.data)
        else:  # only two possible return types
            return [np.sum(element.data) for element in self.traverse_vector(temp)]

    def angle_between(self, other, unit="rad"):
        """ calculate the angle between itself and other"""

        if not isinstance(other, Vector):
            raise TypeError("input must be a Vector: ", other)
        if other.dim != self.dim:
            raise ValueError("Dimension mismatch: ", self.dim, other.dim)
        if unit in ['d', "deg", "degree", "degrees", '°']:
            return np.arccos(self.dot(other)/self.r/other.r) * c.rad2deg
        if unit in ["'", 'min', "minute", "minutes", "m"]:
            return np.arccos(self.dot(other) / self.r / other.r) * c.rad2min
        if unit in ['"', "sec", "second", "seconds", "s"]:
            return np.arccos(self.dot(other) / self.r / other.r) * c.rad2sec

        return np.arccos(self.dot(other)/self.r/other.r)

    def cross(self, other):
        """ define cross product of two Vectors """

        temp = copy.deepcopy(self)
        if isinstance(other, Vector):
            if other.dim != self.dim:
                raise ValueError("Dimension mismatch: ", self.dim, self.dim)
            temp.data = np.cross(temp.data, other.data)
        elif isinstance(other, (np.ndarray, list, tuple)):
            try:
                # use "and" in determination instead of "&" to get the correct result
                if all((isinstance(item, Vector) and item.dim == self.dim) for item in self.traverse_vector(other)):
                    temp = copy.deepcopy(other)
                    # this for loop indeed changes element since it is immutable
                    for element in self.traverse_vector(temp):
                        element.data = np.cross(element.data, self.data)
                        if element.data == np.array(0):
                            element.data = np.zeros(element.dim)
                        element.calculate_r_angles()
                    return temp
            except TypeError:
                # np.asarray(Vector) = Vector.data; but mixed types may confuse asarray
                other = np.asarray([item for item in self.traverse(other)]).flatten()
                if other.size != self.dim:
                    raise ValueError("Dimension mismatch: ", self.dim, other.size)
                temp.data = np.cross(temp.data, other)
        elif isinstance(other, Number):
            raise TypeError("Cannot cross a Vector with a scalar ", other)
        else:
            raise TypeError("Cannot cross a Vector with ", other)

        if temp.data == np.array(0):
            temp.data = np.zeros(temp.dim)
        temp.calculate_r_angles()
        return temp


def minmax(arr):
    """ Print data min/max/range """

    if not isinstance(arr, (np.ndarray, list, tuple)):
        raise TypeError("input should be an array-like object, but it is :", arr)

    print("[min, max]=["+str(np.min(arr))+", "+str(np.max(arr))+"], range="+str(abs(np.max(arr)-np.min(arr))))


def d(y, dx):
    """ calculate the 2nd-order derivative of y(x) on a uniform grid
        N.B.: you may use numpy.gradient, which is more versatile
    """

    if not isinstance(y, (np.ndarray, list, tuple)):
        raise TypeError("y should be an array-like object, but it is :", y)
    if not isinstance(dx, Number):
        raise TypeError("dx should be a scalar, but it is:", dx)

    two_dx = dx * 2
    dydx = np.zeros(y.size)
    dydx[1:-1] = (y[2:] - y[:-2]) / two_dx

    dydx[ 0] = (-   y[ 2] + 4*y[ 1] - 3*y[ 0]) / two_dx
    dydx[-1] = (+ 3*y[-1] - 4*y[-2] +   y[-3]) / two_dx

    return dydx


def fourier_amp(field, k):
    """ calculate the Fourier amplitudes of field on wavenumber k
        :param field: a scalar field
        :param k: which wavenumber to compute
    """

    if not isinstance(field, (np.ndarray, list, tuple)):
        raise TypeError("field should be an array-like object, but it is :", field)

    field = np.asarray(field)
    k = np.asarray(k).flatten()

    if k.size != field.ndim:
        raise ValueError("the size of wavenumber (k) should match the dimension of field:", k.size, field.ndim)

    if field.ndim == 1:
        tmp = np.fft.fft(field)
        return np.sum(abs(tmp[k[0]::field.size-2*k[0]])) / field.size
    elif field.ndim == 2:
        tmp = np.fft.fft2(field)
        if np.all(k == np.zeros(2)):
            return np.sum(abs(tmp[0, 0]))  # AC power
        elif k[0] == 0:
            return np.sum(abs(tmp[0, k[1]::field.shape[1] - 2*k[1]])) / np.prod(field.shape)
        elif k[1] == 0:
            return np.sum(abs(tmp[k[0]::field.shape[0] - 2*k[0], 0])) / np.prod(field.shape)
        else:
            return np.sum(abs(tmp[k[0]::field.shape[0] - 2*k[0], k[1]::field.shape[1] - 2*k[1]])) / np.prod(field.shape)
    else:
        raise RuntimeError("The Fourier amplitude calculation for arrays higher than 2d has not been implemented.")
