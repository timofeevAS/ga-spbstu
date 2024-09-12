import unittest

from utils.number_present import synchronize_binary_strings


class SynchronizeBinaryStringsTests(unittest.TestCase):
    def testcase1(self):
        code1 = '1101'
        code2 = '11'

        code1, code2 = synchronize_binary_strings(code1, code2)

        self.assertEqual(code1,'1101')
        self.assertEqual(code2, '0011')


if __name__ == '__main__':
    unittest.main()
