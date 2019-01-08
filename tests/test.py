""" Test module (this is not a package or subpackage) """

import unittest
import sys
import numpy as np
import copy
sys.path.append("..")
from pyridoxine import utility as rxu

class BasicTestSuite(unittest.TestCase):
    """ Basic test cases. """

    def test_absolute_truth_and_meaning(self):
        """ pass """

        #sys.path.append("..")
        #from pyridoxine import utility

        assert(1 > 0)


    def test_rcParames(self):
        """ pass """

        sys.path.append("..")
        from pyridoxine import plt as rxplt
        import matplotlib.pyplot as plt

        """
        rxplt.plt_params("ppt")
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        rxplt.ax_labeling(ax, x=r"xtitle", y=r"ytitle", t=r"title")
        ax1 = rxplt.add_subplot_axis(ax, [0.2, 0.2, 0.2, 0.2])
        ax1.plot([1, 0], [0, 1])
        ax1.set_facecolor([1, 1, 1, 0])
        cbar = rxplt.add_customized_colorbar(fig, [0, 1], [0.1, 0.05, 0.75, 0.02])
        plt.show()
        plt.close("all")
        """
        assert(1 > 0)


if __name__ == '__main__':
    unittest.main()
