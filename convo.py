from PIL import Image
import numpy as np


# Calculates the gaussian kernel
def make_kernel(ksize, sigma):
    '''
    :param ksize: int kernel size
    :param sigma: int
    :return: 2d numpy array
    '''
    # Initiate an empty kernel and find its center
    kernel = np.zeros((ksize, ksize))
    center = ksize // 2

    '''
    Calculate the kernel values based on the Gaussian equation. G(x, y) = 1 / 2π*sigma2 exp(−(x^2 + y^2)/2sigma^2)
    Where (x, y) is the current pixel position and sigma is the Gaussian parameter.
    x^2+y^2 calculates the distance of the current pixel from the center pixel in the matrix.
    '''
    for i in range(ksize):
        for j in range(ksize):
            distance_squared = (i - center) ** 2 + (j - center) ** 2
            kernel[i, j] = (1 / (2 * np.pi * sigma ** 2)) * np.exp(-distance_squared / (2 * sigma ** 2))

    '''
    Normalize the kernel to ensure the sum equals 1, this is important to ensure the average value of the image is 
    preserved. Otherwise, the brightness might be changed. In the context of Gaussian blurring or smoothing, the kernel 
    matrix represents the weights assigned to neighboring pixels. By normalizing the kernel, the blurring effect is 
    evenly distributed across the image while preserving the average intensity.
    '''

    kernel /= np.sum(kernel)

    return kernel


# Calculate the convolution of the original image and the kernel
def slow_convolve(arr, k):
    '''
    :param arr: input image as np array
    :param k: kernel as np array
    :return: out image as np array
    '''
    # Check if the input image has a color channel
    if len(arr.shape) == 3:
        image_height, image_width, channels = arr.shape
        output_image = np.zeros_like(arr)
    else:
        image_height, image_width = arr.shape
        output_image = np.zeros((image_height, image_width))

    ''' 
    In case the input image isn't symmetrical, the padding has to be adjusted. Otherwise:
    kernel_size = len(k)
    padding = (kernel_size - 1) / 2
    padded_image = np.pad(arr, padding, mode='constant')
    '''
    kernel_height, kernel_width = k.shape
    padding_height = (kernel_height - 1) // 2
    padding_width = (kernel_width - 1) // 2

    if len(arr.shape) == 3:
        # Pad the input with zeros for each channel separately
        padded_image = np.pad(arr, ((padding_height, padding_height), (padding_width, padding_width), (0, 0)), mode='constant')
    else:
        padded_image = np.pad(arr, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

    '''
    Go over each pixel in the original image and calculate the new value using the convolution. It is important to 
    use image_height and width before padding to only iterate over the original image and not the padded border.
    '''
    for i in range(image_height):
        for j in range(image_width):
            # Calculate convolution
            convolution = 0

            # Iterate over u and v, which are the neighboring pixels, and multiply with the corresponding kernel value
            for u in range(-padding_height, padding_height + 1):
                for v in range(-padding_width, padding_width + 1):
                    kernel_value = k[u + padding_height, v + padding_width]
                    if len(arr.shape) == 3:
                        image_value = padded_image[i + padding_height + u, j + padding_width + v, :]
                    else:
                        image_value = padded_image[i + padding_height + u, j + padding_width + v]

                    # Sum up the values for all neighbors and assign it as the new pixel value
                    convolution += kernel_value * image_value

            if len(arr.shape) == 3:
                output_image[i, j, :] = convolution
            else:
                output_image[i, j] = convolution

    return output_image


if __name__ == '__main__':

    # Kernel size and sigma that controls the blurring effect
    k = make_kernel(3, 2)  # todo: find better parameters

    # TODO: choose the image you prefer
    im = np.array(Image.open('input1.jpg'))
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))

    # Convolving the image with the gaussian kernel
    blurred_image = slow_convolve(im, k)

    # Subtracting the result from the original
    unsharp_mask = im - blurred_image
    result = np.clip(im + unsharp_mask, 0, 255).astype(np.uint8)

    # Save the result
    result_image = Image.fromarray(result)
    result_image.save('output.jpg')
