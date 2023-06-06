from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    return  # implement the Gaussian kernel here


def slow_convolve(arr, k):
    return  # implement the convolution with padding here


if __name__ == '__main__':
    k = make_kernel(3, 1)   # todo: find better parameters
    
    # TODO: chose the image you prefer
    # im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
