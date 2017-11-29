""" Test module (this is not a package or subpackage) """

import unittest
import pyridoxine


class BasicTestSuite(unittest.TestCase):
    """ Basic test cases. """

    def test_absolute_truth_and_meaning(self):
        """ pass """

        assert True


if __name__ == '__main__':
    unittest.main()
