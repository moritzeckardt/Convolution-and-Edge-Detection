"""
Created on 22.10.2020
@author: Charly, Max
"""

import unittest
#import cv2
import numpy as np
import os
from convo import slow_convolve, make_kernel
from scipy.signal import convolve

# 5% tolerance
RTOL = 0.001


class TestConvo(unittest.TestCase):

    def test_gauss(self):
        kernel = make_kernel(5, 2)
        self.assertAlmostEqual(1, float(np.sum(kernel)), delta=RTOL, msg="kernel must sum up to one!")
        self.assertTrue(np.allclose(0.0338, kernel[4, 3], rtol=RTOL), msg="Your kernel seems to be wrong")
        self.assertTrue(np.allclose(0.0338, kernel[4, 3], rtol=RTOL), msg="Your kernel seems to be wrong")
        self.assertEqual(kernel[2, 3], kernel[3, 2], msg='kernel must be symmetric!')

    def test_type(self):
        a = np.zeros((10, 10))
        b = np.zeros((3, 3))
        res = slow_convolve(a, b)
        self.assertIsInstance(res, np.ndarray, msg="use numpy arrays")

    def test_size(self):
        for s0 in (3, 5, 11, 99):
            for s1 in (1, 2, 3):
                for s2 in (1, 2, 3):
                    a = np.zeros((s0, s0))
                    b = np.zeros((s1, s2))
                    res = slow_convolve(a, b)
                    self.assertEqual(res.shape, (s0, s0), msg='check the padding')

    def test_flip(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        res = slow_convolve(a, b)
        self.assertEqual(res[0, 0], 5, msg='did you forget to flip the kernel horizontally and vertically?')

    def test_sum(self):
        a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        b = np.array([[2]])
        res = slow_convolve(a, b).sum()
        self.assertEqual(res, 90, msg='check the element-wise operation (multiplication)')
        b = np.array([[1, 1], [2, 2], [3, 3]])
        res = slow_convolve(a, b).sum()
        self.assertEqual(res, 306, msg='check the aggregation operation (addition)')

    def test_compare(self):
        a = np.random.rand(10, 10)
        b = np.random.rand(3, 3)
        res1 = slow_convolve(a, b)
        res2 = convolve(a, b, mode='same')
        diff = np.abs(res1 - res2).sum()
        self.assertTrue(diff <= RTOL,
                        msg='you do not get the same result as scipy.signal.convolve, so there is some error')


if __name__ == '__main__':
    unittest.main()
