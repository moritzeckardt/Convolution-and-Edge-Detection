from PIL import Image
import numpy as np


# Create gaussian kernel according to the formula in the exercise sheet
def make_kernel(ksize, sigma):
    # Initialize the kernel
    kernel = np.zeros((ksize, ksize))

    # Apply the formula for each element in the kernel
    for x in range(ksize):
        for y in range(ksize):
            exponent = -((x - ksize // 2) ** 2 + (y - ksize // 2) ** 2) / (2 * sigma ** 2)
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(exponent)

    # Normalize the kernel -> Ensure that the sum of all elements in the kernel is 1
    kernel /= np.sum(kernel)

    # Return the kernel
    return kernel


# Implement the convolution with padding
def slow_convolve(arr, k):
    return


if __name__ == '__main__':
    # Find best parameters for the kernel (ksize and sigma)
    k = make_kernel(3, 2)
    print(k)

    # Load image
    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
