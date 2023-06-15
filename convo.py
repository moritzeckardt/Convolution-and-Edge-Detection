from PIL import Image
import numpy as np


# Create gaussian kernel according to the formula in the exercise sheet
def make_kernel(ksize, sigma):
    # Initialize the kernel
    kernel = np.zeros((ksize, ksize))

    # Apply the formula for each element in the kernel
    for x in range(ksize):
        for y in range(ksize):
            exponent = -((x - ksize // 2) ** 2 + (y - ksize // 2) ** 2) / (2 * sigma ** 2)  # Distance to center pixel
            kernel[x, y] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(exponent)

    # Normalize the kernel -> Ensure that the sum of all elements in the kernel is 1
    kernel /= np.sum(kernel)

    # Return the kernel
    return kernel


# Implement the convolution with padding
def slow_convolve(arr, k):
    # Get image dimensions
    image_height, image_width = arr.shape

    # Get kernel dimensions
    kernel_height, kernel_width = k.shape

    # Calculate the padding
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # Apply padding to the image without using numpy
    padded_image = np.zeros((image_height + 2 * padding_height, image_width + 2 * padding_width))

    # Create result image
    result_image = np.zeros((image_height, image_width))

    # Flip
    flipped_kernel = np.flipud(np.fliplr(k))

    for i in range(image_height):
        for j in range(image_height):
            for u in range(kernel_height):
                for v in range(kernel_width):
                    result_image[i, j] += flipped_kernel[u, v] * padded_image[i + u, j + v]

    # Return result image
    return result_image


if __name__ == '__main__':
    # Find best parameters for the kernel (ksize and sigma)
    k = make_kernel(3, 2)
    print(k)

    # Test the convolution
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
    slow_convolve(a, b)

    # Load image
    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
