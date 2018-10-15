import sys
import os
import numpy as np
from scipy.misc import imread, imsave, imresize


if __name__ == '__main__':
    assert len(sys.argv) >= 3

    img_name = sys.argv[1]
    plaid_name = sys.argv[2]
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    else:
        threshold = 0.4

    img = imread(img_name) / 255.0
    plaid = imread(plaid_name) / 255.0
    plaid = imresize(plaid, img.shape[:2])

    intensity = np.mean(img, axis=2)
    intensities = np.repeat(intensity[..., np.newaxis], 3, axis=2)
    plaided = img.copy()
    mask = intensities > threshold
    plaided[mask] = plaid[mask] * intensities[mask]

    img_path, _ = os.path.splitext(img_name)
    plaid_path, _ = os.path.splitext(plaid_name)
    _, img_filename = os.path.split(img_path)
    _, plaid_filename = os.path.split(plaid_path)
    out_name = '{}-{}.png'.format(img_filename, plaid_filename)

    imsave(out_name, plaided)
