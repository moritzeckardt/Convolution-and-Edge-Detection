import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from PIL import Image


#
# NO MORE MODULES ALLOWED
#


def gaussFilter(img_in, ksize, sigma):
    """
    filter the image with a gauss kernel
    :param img_in: 2D greyscale image (np.ndarray)
    :param ksize: kernel size (int)
    :param sigma: sigma (float)
    :return: (kernel, filtered) kernel and gaussian filtered image (both np.ndarray)
    """
    # Create a 1D Gaussian kernel
    kernel_1d = np.linspace(-(ksize // 2), ksize // 2, ksize)
    kernel_1d = np.exp(-0.5 * (kernel_1d / sigma) ** 2)
    kernel_1d /= np.sum(kernel_1d)

    # Create a 2D Gaussian kernel by multiplying the 1D kernel with its transpose
    kernel = np.outer(kernel_1d, kernel_1d)

    # Perform convolution using scipy.ndimage.convolve
    filtered = convolve(img_in, kernel)

    return kernel, filtered.astype(int)


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gx = convolve(img_in, kernel_x)
    gy = convolve(img_in, kernel_y)

    return gx.astype(int), gy.astype(int)


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    gradient = np.sqrt(gx ** 2 + gy ** 2)
    theta = np.arctan2(gy, gx)

    return gradient.astype(int), theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    angle = np.rad2deg(angle) % 180
    if angle < 22.5 or angle >= 157.5:
        return 0
    elif angle < 67.5:
        return 45
    elif angle < 112.5:
        return 90
    else:
        return 135


def maxSuppress(g, theta):
    """
    calculate maximum suppression
    :param g:  (np.ndarray)
    :param theta: 2d image (np.ndarray)
    :return: max_sup (np.ndarray)
    """
    rows, cols = g.shape
    suppressed = np.zeros_like(g)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            angle = convertAngle(theta[i, j])

            if angle == 0:
                if g[i, j] >= g[i, j - 1] and g[i, j] >= g[i, j + 1]:
                    suppressed[i, j] = g[i, j]
            elif angle == 45:
                if g[i, j] >= g[i - 1, j + 1] and g[i, j] >= g[i + 1, j - 1]:
                    suppressed[i, j] = g[i, j]
            elif angle == 90:
                if g[i, j] >= g[i - 1, j] and g[i, j] >= g[i + 1, j]:
                    suppressed[i, j] = g[i, j]
            else:
                if g[i, j] >= g[i - 1, j - 1] and g[i, j] >= g[i + 1, j + 1]:
                    suppressed[i, j] = g[i, j]

    return suppressed


def hysteris(max_sup, t_low, t_high):
    """
    calculate hysteris thresholding.
    Attention! This is a simplified version of the lectures hysteresis.
    Please refer to the definition in the instruction

    :param max_sup: 2d image (np.ndarray)
    :param t_low: (int)
    :param t_high: (int)
    :return: hysteris thresholded image (np.ndarray)
    """
    rows, cols = max_sup.shape
    result = np.zeros_like(max_sup)
    strong = max_sup >= t_high
    weak = (max_sup >= t_low) & (max_sup < t_high)

    result[strong] = 255
    visited = set()

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if (i, j) in visited:
                continue

            if weak[i, j]:
                visited.add((i, j))
                neighbors = [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1),
                             (i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]

                for neighbor in neighbors:
                    ni, nj = neighbor
                    if (ni, nj) not in visited and strong[ni, nj]:
                        result[i, j] = 255
                        visited.add((i, j))
                        break

    return result


def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 3)

    # sobel
    gx, gy = sobel(gauss)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(gx, 'gray')
    plt.title('gx')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(gy, 'gray')
    plt.title('gy')
    plt.colorbar()
    plt.show()

    # gradient directions
    g, theta = gradientAndDirection(gx, gy)

    # plotting
    plt.subplot(1, 2, 1)
    plt.imshow(g, 'gray')
    plt.title('gradient magnitude')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(theta)
    plt.title('theta')
    plt.colorbar()
    plt.show()

    # maximum suppression
    maxS_img = maxSuppress(g, theta)

    # plotting
    plt.imshow(maxS_img, 'gray')
    plt.show()

    result = hysteris(maxS_img, 50, 75)

    return result

if __name__ == "__main__":
    img = np.array(Image.open('contrast.jpg').convert('L'))
    canny(img)
