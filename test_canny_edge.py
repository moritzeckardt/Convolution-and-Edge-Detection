"""
Created on 22.10.2020
@author: Charly, Max
"""

import unittest
import cv2
import numpy as np
#from CannyEdgeDetector_solution import gaussFilter, sobel, gradientAndDirection, maxSuppress, hysteris, convertAngle
from CannyEdgeDetector import gaussFilter

# 5% tolerance
RTOL = 0.05


class TestCanny(unittest.TestCase):

    def setUp(self) -> None:
        self.img = cv2.imread('contrast.jpg', cv2.IMREAD_GRAYSCALE)
        assert self.img.dtype == np.uint8

    # test canny edge tasks
    def test_gauss(self):
        kernel, res = gaussFilter(self.img, 5, 2)
        self.assertIsInstance(res, np.ndarray, msg='Use numpy arrays')
        self.assertTrue(res.dtype == int, msg='cast filtered image back to int')
        self.assertAlmostEqual(1, float(np.sum(kernel)), delta=RTOL, msg="kernel must sum up to one!")
        self.assertTrue(np.allclose(0.0338, kernel[4, 3], rtol=RTOL), msg="Your kernel seems to be wrong")
        self.assertTrue(np.allclose(0.0338, kernel[4, 3], rtol=RTOL), msg="Your kernel seems to be wrong")
        self.assertEqual(kernel[2, 3], kernel[3, 2], msg='kernel must be symmetric!')
        self.assertTrue(np.abs(res[100, 100] - 165) <= 3)
        self.assertTrue(np.abs(res[400, 400] - 129) <= 3)

    def test_sobel(self):
        img_in = np.zeros((9, 9))
        img_in[4] = 1
        img_in[:, 4] = 1
        gx, gy = sobel(img_in)
        # numpy array (5,5) with values from 0 to 32 in 2 steps and calculate the sum of each result array
        self.assertIsInstance(gx, np.ndarray, msg='Use numpy arrays')
        self.assertIsInstance(gy, np.ndarray, msg='Use numpy arrays')
        self.assertTrue(gx.dtype == int, msg='cast filtered image back to int')
        self.assertEqual(-4, gx[1, 3], msg='check your sobel x filter. Image coordinate System!')
        self.assertEqual(4, gx[7, 5], msg='check your sobel x filter. Image coordinate System!')
        self.assertEqual(-2, gx[4, 3], msg='check your sobel x filter. Image coordinate System!')
        self.assertEqual(4, gy[3, 1], msg='check your sobel y filter. Image coordinate System!')
        self.assertEqual(-4, gy[5, 7], msg='check your sobel y filter. Image coordinate System!')
        self.assertEqual(2, gy[3, 4], msg='check your sobel y filter. Image coordinate System!')

    def test_gradient_direction(self):
        g, t = gradientAndDirection(np.array([[-13, 15], [-12, 11]]), np.array([[15, 21], [4, 1]]))
        self.assertIsInstance(g, np.ndarray, msg='Use numpy arrays')
        self.assertIsInstance(t, np.ndarray, msg='Use numpy arrays')
        self.assertTrue(g.dtype == int, msg='cast filtered image back to int')
        self.assertTrue(t.dtype == float, msg='angles in radias should be floating point precision')
        self.assertTrue(np.all(g == np.array([[19, 25], [12, 11]])) or np.all(g == np.array([[20, 26], [13, 11]])))
        self.assertTrue(np.allclose(t, np.array([[2.28, 0.95], [2.81, 0.09]]), rtol=RTOL))

    def test_convertAngle(self):
        self.assertIsInstance(convertAngle(175), int, msg='Degrees are ints!')
        self.assertEqual(45, convertAngle(0.393))
        self.assertEqual(0, convertAngle(0.392))
        self.assertEqual(90, convertAngle(1.96349))
        self.assertEqual(0, convertAngle(3.12929))
        self.assertEqual(90, convertAngle(1.56079))
        self.assertEqual(135, convertAngle((112.5 / 360) * 2 * np.pi))
        self.assertEqual(0, convertAngle((157.5 / 360) * 2 * np.pi))
        self.assertEqual(135, convertAngle(8.246680715673207))

    def test_maxSuppress(self):
        # TODO winkel explizit abfragen
        g = np.array([[12, 33, 55, 7],
                      [4, 66, 34, 14],
                      [98, 34, 6, 72],
                      [23, 44, 4, 37]])
        g2 = np.array([[12, 77, 55, 7],
                       [4, 34, 35, 14],
                       [98, 34, 50, 72],
                       [23, 44, 4, 37]])
        t = np.array([[212, 433, 655, 547],
                      [423, 1.57, 0.01, 145],
                      [985, 0.8, 2.1, 772],
                      [236, 444, 422, 374]])
        self.assertIsInstance(maxSuppress(g.copy(), t.copy()), np.ndarray, msg='Use numpy arrays')
        # self.assertEqual(maxSuppress(g.copy(), t.copy())[0, 0], 0, msg="use zero padding!")

        self.assertEqual(maxSuppress(g.copy(), t.copy())[1, 1], 66, msg='check 90 deg')
        self.assertEqual(maxSuppress(g2.copy(), t.copy())[1, 1], 0, msg='check 90 deg')

        self.assertEqual(maxSuppress(g.copy(), t.copy())[2, 2], 0, msg='check 135 deg')
        self.assertEqual(maxSuppress(g2.copy(), t.copy())[2, 2], 50, msg='check 135 deg')

        self.assertEqual(maxSuppress(g.copy(), t.copy())[2, 1], 34, msg='check 45 deg (>=)')
        self.assertEqual(maxSuppress(g2.copy(), t.copy())[2, 1], 0, msg='check 45 deg')

        self.assertEqual(maxSuppress(g.copy(), t.copy())[1, 2], 0, msg='check 0 deg')
        self.assertEqual(maxSuppress(g2.copy(), t.copy())[1, 2], 35, msg='check 0 deg')

    def test_hysteris(self):
        # FIXME: wenn nur 2 = 255 läuft der Test trotzdem durch (keine Abfrage von 1 notwendig) -.-
        test_array = np.array([[4, 4, 2, 5, 3, 1],
                               [1, 5, 2, 7, 8, 1],
                               [1, 4, 1, 3, 4, 1],
                               [1, 7, 2, 6, 13, 1]])
        res = hysteris(test_array.copy(), 3, 6)
        res2 = hysteris(test_array.copy(), 3, 11)
        self.assertIsInstance(res, np.ndarray, msg='Use numpy arrays')
        whites = len(np.where(res == 255)[0])
        whites2 = len(np.where(res2 == 255)[0])
        # allowing border inclusion and exclusion
        self.assertIn(whites, [4, 8])
        self.assertIn(whites2, [1, 3])


if __name__ == '__main__':
    unittest.main()
