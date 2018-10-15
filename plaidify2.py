import sys
import os
import numpy as np
from scipy.misc import imread, imsave, imresize
from scipy.ndimage import convolve, gaussian_filter
import matplotlib.pyplot as plt


def get_vertical_gradient_magnitudes(img, kernel_size):
    kernel_shape = [kernel_size * 2 - 1, 1]
    kernel = np.arange(-kernel_size + 1, kernel_size).reshape(kernel_shape)
    return convolve(img, kernel)


def get_horizontal_gradient_magnitudes(img, kernel_size):
    return get_vertical_gradient_magnitudes(img.T, kernel_size).T


def get_gradient_magnitudes(img):
    x = get_horizontal_gradient_magnitudes(img)
    y = get_vertical_gradient_magnitudes(img)
    return np.sqrt(x ** 2 + y ** 2)


def get_gradient_directions(img):
    x = get_horizontal_gradient_magnitudes(img)
    y = get_vertical_gradient_magnitudes(img)
    return np.arctan(y, x)


def move_pixels(intensity, mapped, scale, kernel_size, blur):
    blurred = gaussian_filter(intensity, blur)
    dx = get_horizontal_gradient_magnitudes(blurred, kernel_size) * scale
    dy = get_vertical_gradient_magnitudes(blurred, kernel_size) * scale
    height, width = intensity.shape
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x_shifted = np.round(x + dx)
    x_shifted = np.clip(x_shifted, 0, width - 1).astype(np.int)
    y_shifted = np.round(y + dy)
    y_shifted = np.clip(y_shifted, 0, height - 1).astype(np.int)
    return mapped[y_shifted, x_shifted, :]


def show_quiver(img, kernel_size):
    x = get_horizontal_gradient_magnitudes(img, kernel_size)
    y = get_vertical_gradient_magnitudes(img, kernel_size)
    u = 150
    d = 200
    plt.imshow(img[u:d, u:d])
    plt.quiver(x[u:d, u:d], -y[u:d, u:d])
    plt.show()


if __name__ == '__main__':
    assert len(sys.argv) >= 3

    img_name = sys.argv[1]
    plaid_name = sys.argv[2]

    if len(sys.argv) > 3:
        scale = int(sys.argv[3])
    else:
        scale = 50

    if len(sys.argv) > 4:
        blur = int(sys.argv[4])
    else:
        blur = 5

    if len(sys.argv) > 5:
        kernel_size = int(sys.argv[5])
    else:
        kernel_size = 3

    img = imread(img_name) / 255.0
    plaid = imread(plaid_name) / 255.0
    plaid = imresize(plaid, img.shape[:2])

    intensity = np.mean(img, axis=2)
    plaided = move_pixels(intensity, plaid, scale, kernel_size, blur) * img

    img_path, _ = os.path.splitext(img_name)
    plaid_path, _ = os.path.splitext(plaid_name)
    _, img_filename = os.path.split(img_path)
    _, plaid_filename = os.path.split(plaid_path)
    out_name = '{}-{}.png'.format(img_filename, plaid_filename)

    imsave(out_name, plaided)
