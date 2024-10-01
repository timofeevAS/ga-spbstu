import unittest

from utils.number_present import in_range


class InRangeTests(unittest.TestCase):
    def testcase1(self):
        point = 5.0
        num_range = (-3.5, 5.0)
        self.assertTrue(in_range(point, num_range))

    def testcase2(self):
        point = 5.0
        num_range = (5.01, 20)
        self.assertFalse(in_range(point, num_range))

if __name__ == '__main__':
    unittest.main()
