import os.path
import pathlib

import matplotlib.pyplot as plt
import numpy as np
from skimage.exposure import rescale_intensity
from skimage.io import imread


def contrast_stretch(img):
    v_min, v_max = np.percentile(img, (0.2, 98.0))
    return rescale_intensity(img, in_range=(v_min, v_max))


def read_images_4(dir_path, imgs):
    for _img in imgs:

        # Set figure
        fig = plt.figure(_img)
        rows = 2
        columns = 2

        # Read image
        img = imread(os.path.join(dir_path, _img))

        # Add first image to figure
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img)
        plt.axis("off")

        # Add contrasted image to figure
        fig.add_subplot(rows, columns, 2)
        img2 = contrast_stretch(img)
        plt.imshow(img2)
        plt.axis("off")

        # Add histogram for first image
        fig.add_subplot(rows, columns, 3)
        plt.hist(img.ravel(), bins=50)

        # Add histogram for second image
        fig.add_subplot(rows, columns, 4)
        plt.hist(img2.ravel(), bins=50)

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join("images", "../input1")
    read_images_4(dir_path, os.listdir(dir_path))
