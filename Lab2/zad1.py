import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imsave, imread


def add_noise(noise_type, image):
    if noise_type == "gauss":
        shp = image.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, shp)
        noise = image + gauss
        return noise

    if noise_type == "uniform":
        k = 0.5

        if image.ndim == 2:
            row, col = image.shape
            noise = np.random.rand(row, col)
        else:
            row, col, chan = image.shape
            noise = np.random.rand(row, col, chan)

        noise = image + k * (noise - 0.5)

        return noise

    if noise_type == "s&p":
        sh = image.shape
        s_vs_p = 0.5
        amount = 0.05
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in sh]
        out[coords] = 1

        num_peper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_peper)) for i in sh]
        out[coords] = 0
        return out

    if noise_type == "poisson":
        PEAK = 20
        noisy = image + np.random.poisson(0.5 * PEAK, image.shape) / PEAK
        return noisy

    if noise_type == "speckle":
        if image.ndim == 2:
            row, col = image.shape
            noise = np.random.randn(row, col)
        else:
            row, col, chan = image.shape
            noise = np.random.randn(row, col, chan)

        noisy = image + image * 0.5 * noise
        return noisy


def read_images_1(img_path):

    row, column = 3, 2
    # Original picture
    fig = plt.figure(img_path)
    plt.axis("off")
    img = imread(img_path)

    fig.add_subplot(row, column, 1, title="Original")
    plt.imshow(img, cmap="gray")
    plt.axis("off")

    fig.add_subplot(row, column, 2, title="Gauss")
    plt.imshow(add_noise("gauss", img), cmap="gray")
    plt.axis("off")

    fig.add_subplot(row, column, 3, title="Uniform")
    plt.imshow(add_noise("uniform", img), cmap="gray")
    plt.axis("off")

    fig.add_subplot(row, column, 4, title="s&p")
    plt.imshow(add_noise("s&p", img), cmap="gray")
    plt.axis("off")

    fig.add_subplot(row, column, 5, title="Poisson")
    plt.imshow(add_noise("poisson", img), cmap="gray")
    plt.axis("off")

    fig.add_subplot(row, column, 6, title="Speckle")
    plt.imshow(add_noise("speckle", img), cmap="gray")
    plt.axis("off")

    plt.show(block=True)


if __name__ == "__main__":
    img_path = os.path.join(pathlib.Path(__file__).parent.parent, "input1", "coins.png")
    read_images_1(img_path)
