""" Provide constants used by Pyridoxine """

import numpy as np


class Constants:
    """ Constants used by Pyridoxine """

    def __init__(self):

        self.pi = np.pi
        self.rad2deg = 180. / np.pi
        self.rad2min = self.rad2deg * 60.
        self.rad2sec = self.rad2min * 60.


c = Constants()