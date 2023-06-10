from PIL import Image
import numpy as np


# Implement the Gaussian kernel
def make_kernel(ksize, sigma):
    # Create kernel according to the formula in the exercise sheet
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        -((x - ksize // 2) ** 2 + (y - ksize // 2) ** 2) / (2 * sigma ** 2)), (ksize, ksize))

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Return the kernel
    return kernel


# Implement the convolution with padding
def slow_convolve(arr, k):
    image_height, image_width = arr.shape
    kernel_height, kernel_width = k.shape

    padding_height = kernel_height // 2
    padding_width = kernel_width // 2

    padded_image = np.pad(arr, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

    result = np.zeros_like(arr)

    # Flip the kernel horizontally and vertically
    flipped_kernel = np.flipud(np.fliplr(k))

    for i in range(image_height):
        for j in range(image_width):
            for u in range(kernel_height):
                for v in range(kernel_width):
                    result[i, j] += flipped_kernel[u, v] * padded_image[i + u, j + v]

    return result


if __name__ == '__main__':
    # Find best parameters for the kernel (ksize and sigma)
    k = make_kernel(5, 5/2)
    print(k)

    # Load image
    im = np.array(Image.open('input1.jpg').convert('L'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    result = im + (im - slow_convolve(im, k))
    result = np.clip(result, 0, 255)

    image = Image.fromarray(result.astype(np.uint8))
    image.show()


    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
