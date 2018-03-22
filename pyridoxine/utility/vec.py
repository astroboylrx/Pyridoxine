""" Provide vector classes and related calculations """

import numpy as np
import copy
from numbers import Number
import warnings


class Vector:
    """ Define a vector class for vector calculation. """

    def __init__(self, *args):
        """
        Construct a vector (currently only support 2D and 3D)
        :param args: x1, x2[, x3] or numpy.ndarray or list
        """

        if len(args) is 0:
            raise SyntaxError("Vector must take [an] argument[s] to proceed.")

        self.data = np.array([])

        for i in range(len(args)):
            if isinstance(args[i], np.ndarray):
                self.data = np.append(self.data, (args[i]).flatten())
            elif isinstance(args[i], list):
                self.data = np.append(self.data, np.asarray(args[i]).flatten())
            elif isinstance(args[i], Number):
                self.data = np.append(self.data, args[i])
            else:
                raise TypeError("Cannot establish a vector with non-numeric argument: ", args[i])

        self.dim = self.data.size

        if self.dim is 0: # should not happen
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

    def calculate_r_angles(self):
        """ Compute |r| and angles once data has changed. """

        self.r = np.sqrt(np.sum(self.data ** 2))
        self.phi = np.arctan2(self.data[1], self.data[0])
        if self.dim is 3:
            self.theta = np.arccos(self.data[2] / self.r)

    def __add__(self, other):
        """ Overload the return value of self + other """

        temp = copy.deepcopy(self)
        if isinstance(other, Vector):
            if other.dim != temp.dim:
                raise ValueError("Dimension mismatch: ", self.dim, other.dim)
            temp.data += other.data
        elif isinstance(other, (np.ndarray, list)):
            other = np.asarray(other).flatten()
            if other.size != self.dim:
                raise ValueError("Dimension mismatch: ", self.dim, len(other))
            temp.data += other
        elif isinstance(other, Number):
            temp.data += other
        else:
            raise TypeError("Cannot add ", other, " to Vector ", self.data)

        temp.calculate_r_angles()
        return temp

    def __radd__(self, other):
        """ Overload the return value of other + self """

        return self.__add__(other)

    def __iadd__(self, other):
        """ Overload self += other. """

        return self.__add__(other)

    def __mul__(self, other):
        """ Overload the return value of dot product: self * other """

        temp = copy.deepcopy(self)
        if isinstance(other, Vector):
            if other.dim != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
            temp.data *= other.data
        elif isinstance(other, (np.ndarray, list)):
            other = np.asarray(other).flatten()
            if other.size != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, len(other))
            temp.data *= other
        elif isinstance(other, Number):
            temp.data *= other
        else:
            raise TypeError("Cannot multiply ", other, " to Vector ", self.data)

        temp.calculate_r_angles()
        return temp

    def __rmul__(self, other):
        """ Overload the return value of dot product: other * self """

        return self.__mul__(other)

    def __imul__(self, other):
        """ Overload self *= other """

        return self.__mul__(other)

    def __sub__(self, other):
        """ Overload the return value of self - other """

        if isinstance(other, (Vector, np.ndarray, Number)):
            other *= -1
        elif isinstance(other, list):
            other = [x * (-1) for x in other]
        else:
            raise TypeError("Cannot subtract ", other, " from Vector ", self.data)
        return self.__add__(other)

    def __rsub__(self, other):
        """ Overload the return value of other - self """

        if not isinstance(other, (Vector, np.ndarray, list, Number)):
            raise TypeError("Cannot subtract Vector ", self.data, " from ", other)

        temp = copy.deepcopy(self)
        temp *= (-1)
        return self.__add__(other)

    def __isub__(self, other):
        """ Overload self -= other """

        return self.__sub__(other)

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
        elif isinstance(other, np.ndarray):
            warnings.warn("""Dividing Vector by numpy.ndarray (element by element)""")
            other = np.asarray(other).flatten()
            if other.size != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, len(other))
            return self.__mul__(1. / other)
        elif isinstance(other, list):
            warnings.warn("""Dividing Vector by list (element by element)""")
            other = np.asarray(other).flatten()
            if other.size != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, len(other))
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
        elif isinstance(other, np.ndarray):
            warnings.warn("""Dividing numpy.ndarray by Vector (element by element)""")
            other = np.asarray(other).flatten()
            if other.size != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, len(other))
            return Vector(other / self.data)
        elif isinstance(other, list):
            warnings.warn("""Dividing list by Vector (element by element)""")
            other = np.asarray(other).flatten()
            if other.size != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, len(other))
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

    def cross(self, other):
        """ define cross product of two Vectors """

        if isinstance(other, Vector):
            if other.dim != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, other.dim)
            if self.is_parallel(other):
                return Vector(np.zeros(self.dim))
            return Vector(np.cross(self.data, other.data))
        elif isinstance(other, (np.ndarray, list)):
            other = np.asarray(other).flatten()
            if other.size != self.dim:
                raise ValueError("Dimensions mismatch: ", self.dim, other.size)
            if self.is_parallel(Vector(other)):
                return Vector(np.zeros(self.dim))
            return Vector(np.cross(self.data, other))
        else:
            raise TypeError("Cannot cross product Vector ", self.data, " with ", other)
