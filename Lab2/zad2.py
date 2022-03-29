import os
import pathlib

import imageio
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from skimage import color, img_as_ubyte, data, img_as_int
from skimage import filters
from skimage.filters.rank import mean, mean_percentile, mean_bilateral, median
from skimage.io import imread
from skimage.morphology import disk


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

        noisy = image + image * 0.2 * noise
        return noisy


def lowpass_filter(filter_type, image):
    if filter_type == "gauss":
        if image.ndim > 2:
            image = color.rgb2gray(image)
        return filters.gaussian(image, sigma=2, mode="reflect")

    if filter_type == "mean":
        image = image.astype(np.uint16)
        return mean(img_as_ubyte(image), footprint=disk(20))

    if filter_type == "mean_bilateral":
        image = image.astype(np.uint16)
        return mean_bilateral(image, footprint=disk(20), s0=50, s1=50)

    if filter_type == "median":
        image = image.astype(np.uint16)
        return median(img_as_ubyte(image), footprint=disk(2))


def read_images_2(img_path):

    row, column = 4, 2
    # Original picture
    fig = plt.figure(img_path)
    plt.axis("off")
    img = imageio.imread(img_path)
    i = 1
    fig.add_subplot(row, column, i, title="Gauss Noise")
    gauss_noise = add_noise("gauss", img)
    plt.imshow(gauss_noise, cmap="gray")
    plt.axis("off")
    i += 1
    fig.add_subplot(row, column, i, title="Filtered Gauss")
    plt.imshow(lowpass_filter("gauss", gauss_noise), cmap="gray")
    plt.axis("off")
    i += 1

    fig.add_subplot(row, column, i, title="Uniform Noise")
    uniform = add_noise("uniform", img)
    plt.imshow(uniform, cmap="gray")
    plt.axis("off")
    i += 1

    fig.add_subplot(row, column, i, title="Filtered Mean")
    plt.imshow(lowpass_filter("mean", uniform), cmap="gray")
    plt.axis("off")
    i += 1

    fig.add_subplot(row, column, i, title="S&P Noise")
    s_and_p = add_noise("s&p", img)
    plt.imshow(s_and_p, cmap="gray")
    plt.axis("off")
    i += 1

    fig.add_subplot(row, column, i, title="Filtered Median")
    plt.imshow(lowpass_filter("median", s_and_p), cmap="gray")
    plt.axis("off")
    i += 1

    fig.add_subplot(row, column, i, title="Speckle Noise")
    speckle = add_noise("speckle", img)
    plt.imshow(speckle, cmap="gray")
    plt.axis("off")
    i += 1

    fig.add_subplot(row, column, i, title="Filtered Mean Bilateral")
    plt.imshow(lowpass_filter("mean_bilateral", speckle), cmap="gray")
    plt.axis("off")

    plt.show(block=True)


if __name__ == "__main__":
    img_path = os.path.join(pathlib.Path(__file__).parent.parent, "input1", "coins.png")
    read_images_2(img_path)
