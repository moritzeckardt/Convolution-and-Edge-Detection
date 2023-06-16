import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import convolve


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
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x-ksize // 2) ** 2 + (y - ksize // 2) ** 2) / (2 * sigma ** 2)), (ksize, ksize))
    
    #Normalize kernel
    kernel /= np.sum(kernel)

    filtered_img = convolve(img_in, kernel)

    return kernel, filtered_img.astype(int)


def sobel(img_in):
    """
    applies the sobel filters to the input image
    Watch out! scipy.ndimage.convolve flips the kernel...

    :param img_in: input image (np.ndarray)
    :return: gx, gy - sobel filtered images in x- and y-direction (np.ndarray, np.ndarray)
    """
    sobel_x = np.array([[-1, 0, 1], 
                        [-2, 0, 2],
                        [-1, 0, 1]])

    sobel_y = np.array([[-1, -2, -1], 
                        [0, 0, 0], 
                        [1, 2, 1]])

    gx = convolve(img_in, sobel_x)
    gy = convolve(img_in, sobel_y)

    return gx, gy


def gradientAndDirection(gx, gy):
    """
    calculates the gradient magnitude and direction images
    :param gx: sobel filtered image in x direction (np.ndarray)
    :param gy: sobel filtered image in x direction (np.ndarray)
    :return: g, theta (np.ndarray, np.ndarray)
    """
    g = np.sqrt(gx**2 + gy**2)

    theta = np.arctan2(gx, gy)

    return g.astype(int), theta


def convertAngle(angle):
    """
    compute nearest matching angle
    :param angle: in radians
    :return: nearest match of {0, 45, 90, 135}
    """
    angle = np.rad2deg(angle) % 180
    if angle <= 22.5 or angle >= 157.5:
        return 0
    elif angle < 67.5:
        return 46
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
    x, y = g.shape
    max_sup = np.zeros_like(g)

    for i in range(1, x - 1):
        for j in range(1, y - 1):
            angle = convertAngle(theta[i, j])
            if angle == 0:
                if g[i, j] >= g[i, j -1] and g[i, j] >= g[i, j +1]:
                    max_sup[i, j] = g[i, j]
            elif angle == 45:
                if g[i, j] >= g[i + 1, j - 1] and g[i, j] >= g[i - 1, j + 1]:
                    max_sup[i, j] = g[i, j]
            elif angle == 90:
                if g[i, j] >= g[i - 1, j] and g[i, j] >= g[i + 1, j]:
                    max_sup[i, j] = g[i, j]
            else:
                if g[i, j] >= g[i - 1, j -1] and g[i, j] >= g[i + 1, j +1]:
                    max_sup[i, j] = g[i, j]

    return max_sup

img = np.array(Image.open('contrast.jpg').convert('L'))
img_blurr, gauss_kernel = gaussFilter(img, 5, 2)
sobel_x, sobel_y = sobel(img_blurr)
gradient_mag, theta = gradientAndDirection(sobel_x, sobel_y)
surpressed = maxSuppress(gradient_mag, theta)

image = Image.fromarray(surpressed)
image.show()


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
    # TODO
    pass



def canny(img):
    # gaussian
    kernel, gauss = gaussFilter(img, 5, 2)

    # sobel
    gx, gy = sobel(gauss)

    #Gradient magnitude
    magnitude = gradientAndDirection(gx, gy)

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
