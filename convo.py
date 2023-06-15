from PIL import Image
import numpy as np


# Create gaussian kernel according to the formula in the exercise sheet
def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize), np.float32)
    center = ksize // 2

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

    # Calculate padding
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    # Flip the kernel (Asked for allowance)
    flipped_kernel = np.flipud(np.fliplr(k))

    # Apply padding to the image without using numpy
    padded_image = np.zeros((image_height + 2 * padding_height, image_width + 2 * padding_width))
    padded_image[padding_height:padding_height + image_height, padding_width:padding_width + image_width] = arr

    # Create convolved image
    convolved_image = np.zeros((image_height, image_width))

    # Apply convolution
    for i in range(image_height):
        for j in range(image_height):
            for u in range(kernel_height):
                for v in range(kernel_width):
                    convolved_image[i, j] += flipped_kernel[u, v] * padded_image[i + u, j + v]

    # Return the convolved image
    return convolved_image


if __name__ == '__main__':
    # Find best parameters for the kernel (ksize and sigma)
    k = make_kernel(9, 2)

    # Test convolution
    a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    b = np.array([[2]])
    test_image = slow_convolve(a, b)

    # Load image
    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg').convert('L'))
    # im = np.array(Image.open('input3.jpg').convert('L'))

    # Convolve image
    conv_img = slow_convolve(im, k)

    # Apply sharpening and show image
    result = im + (im - conv_img)  # Blur the image, subtract the result to the input, add the result to the input
    result = np.clip(result, 0, 255)  # Range [0,255] (remember warme-up exercise?)
    image = Image.fromarray(result.astype(np.uint8))  # Convert the array to np.unit8, and save the result
    image.show()

