from PIL import Image
import numpy as np


def make_kernel(ksize, sigma):
    kernel = np.zeros((ksize, ksize), np.float32)
    center = ksize//2

    for i in range(ksize):
        for j in range(ksize):
            x = i -center
            y = j - center
            exponent = -(x**2 + y**2) / (2 * sigma**2)
            kernel[i, j] = (1 / 2 * np.pi * sigma**2) * np.exp(exponent)
    
    return kernel // np.sum(kernel)


def slow_convolve(arr, k):
    #Zu grayscale ändern 
    img = np.dot(arr, [0.2989, 0.5870, 0.1140])
    padding_width = len(k) // 2
    img_height, img_width = img.shape
    ksize = k.shape[0]

    #Bild mit zero padding
    img_padding = np.pad(img, padding_width, mode='constant')
    #img_padding[padding_width:-padding_width, padding_width:-padding_width] = img

    convolved_img = np.zeros(shape=(img_height, img_width))

    for i in range(img_height):
        for j in range(img_width):
            image_matrix = img_padding[i:i+ksize, j:j+ksize]
            convolved_img[i, j] = np.sum(image_matrix * k)
    
    return convolved_img


if __name__ == '__main__':
    k = make_kernel(3, 1)   # todo: find better parameters
    
    # TODO: chose the image you prefer
    im = np.array(Image.open('input1.jpg'))
    conv_img = slow_convolve(im, k)
    image = Image.fromarray(conv_img.astype(np.uint8))
    image.show()
    # im = np.array(Image.open('input2.jpg'))
    # im = np.array(Image.open('input3.jpg'))
    
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
