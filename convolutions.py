import argparse

import cv2
import numpy as np
from skimage.exposure import rescale_intensity


def convolve(image, kernel):
    image_height, image_width = image.shape[:2]
    kernel_height, kernel_width = kernel.shape[:2]

    pad = (kernel_width - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    output = np.zeros((image_height, image_width), dtype='float')

    for y in range(pad, image_height + pad):
        for x in range(pad, image_width + pad):
            roi = image[y - pad: y + pad + 1, x - pad: x + pad + 1]

            k = (roi * kernel).sum()

            output[y - pad, x - pad] = k

    output = rescale_intensity(output, in_range=(0, 255))
    output = (output * 255).astype('uint8')

    return output


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-i', '--image', required=True, help='Path to the input image.')
arguments = vars(argument_parser.parse_args())

# Kernels
small_blur = np.ones((7, 7), dtype='float') * (1.0 / (7 * 7))
large_blur = np.ones((21, 21), dtype='float') * (1.0 / (21 * 21))
sharpen = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]], dtype='int')
laplacian = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]], dtype='int')
sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]], dtype='int')
sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]], dtype='int')
emboss = np.array([
    [-2, -1, 0],
    [-1, 1, 1],
    [0, 1, 2]], dtype='int')

kernel_bank = [
    ('small_blur', small_blur),
    ('large_blur', large_blur),
    ('sharpen', sharpen),
    ('laplacian', laplacian),
    ('sobel_x', sobel_x),
    ('sobel_y', sobel_y),
    ('emboss', emboss)
]

image = cv2.imread(arguments['image'])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

for kernel_name, kernel in kernel_bank:
    print(f'[INFO] Applying {kernel_name} kernel.')
    convolve_output = convolve(gray, kernel)
    opencv_output = cv2.filter2D(gray, -1, kernel)

    # Show images
    cv2.imshow('Original', gray)
    cv2.imshow(f'{kernel_name} convolve', convolve_output)
    cv2.imshow(f'{kernel_name} OpenCV', opencv_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
