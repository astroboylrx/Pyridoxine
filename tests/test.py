""" Test module (this is not a package or subpackage) """

import unittest
import sys
import numpy as np


class BasicTestSuite(unittest.TestCase):
    """ Basic test cases. """

    def test_absolute_truth_and_meaning(self):
        """ pass """

        sys.path.append("..")
        from pyridoxine import utility

        a = utility.Vector([2, 3])
        print(a + 2, a - 2, a * 2, a / 2)
        print(2 + a, 2 - a, 2 * a)
        a[0] = 4
        a.show()

        b = utility.Vector([4], np.array([6]))
        print(a + b, a - b, a * b, a / b)

        b = [4, 6]
        print(a + b, a - b, a * b, a / b)

        print(a.cross(b))

        assert(a.r > 1)

    def test_rcParames(self):
        """ pass """

        sys.path.append("..")
        from pyridoxine import plt as rxplt

        import matplotlib.pyplot as plt

        plt.rcParams.update(rxplt.plt_params("ppt"))
        fig, ax = plt.subplots()
        ax.plot([0, 1], [0, 1])
        rxplt.ax_labeling(ax, x=r"xtitle", y=r"ytitle", t=r"title")
        plt.show()
        plt.close("all")


if __name__ == '__main__':
    unittest.main()
