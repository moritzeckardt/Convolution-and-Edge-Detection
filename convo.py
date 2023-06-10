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

    kernel /= np.sum(kernel)
    return kernel


def slow_convolve(arr, k):
    #Zu grayscale ändern 
    #img = np.dot(arr, [0.2989, 0.5870, 0.1140]) 
    img_height, img_width = arr.shape #Höhe und Breite des Bildes
    kernel_height, kernel_width = k.shape
    padding = kernel_height // 2

    output_height = img_height + 2 * padding - kernel_height + 1 
    output_width = img_width + 2 * padding - kernel_width + 1


    flipped_kernel = np.flipud(np.fliplr(k))

    convolved_img = np.zeros_like(arr)

    #Bild mit zero padding
    img_padding = np.pad(arr, padding, mode='constant')
    #img_padding[padding_width:-padding_width, padding_width:-padding_width] = img


    for i in range(img_height):
        for j in range(img_width):
            for u in range(kernel_height):
                for v in range(kernel_width):
                    convolved_img[i, j] += flipped_kernel[u, v] * img_padding[i + u, j + v]
            
            
            #image_matrix = img_padding[i:i+kernel_height, j:j+kernel_width]
            #convolved_img[i, j] = np.sum(np.dot(image_matrix, k))
    
    return convolved_img


if __name__ == '__main__':
    k = make_kernel(9, 9/2)   # todo: find better parameters
    
    # TODO: chose the image you prefer
    #im = np.array(Image.open('input1.jpg'))
    im = np.array(Image.open('input2.jpg').convert('L'))
    #im = np.array(Image.open('input3.jpg'))

    conv_img = slow_convolve(im, k)
    
    result = im + (im - conv_img)
    result = np.clip(result, 0, 255)
    image = Image.fromarray(result.astype(np.uint8))
    image.show()
   
    # TODO: blur the image, subtract the result to the input,
    #       add the result to the input, clip the values to the
    #       range [0,255] (remember warme-up exercise?), convert
    #       the array to np.unit8, and save the result
