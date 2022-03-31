import os.path
import pathlib
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import gaussian
from skimage.filters.rank import mean, mean_bilateral, median
from skimage.morphology import disk


class Noise(Enum):
    gauss = "gauss"
    uniform = "uniform"
    s_n_p = "s&p"
    poisson = "poisson"
    speckle = "speckle"


class Filter(Enum):
    median = "median"
    gauss = "gauss"
    mean = "mean"
    mean_bilateral = "mean_bilateral"


def add_noise(noise_type, image):
    if noise_type == Noise.gauss:
        shp = image.shape
        mean = 0
        var = 0.01
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, shp)
        noise = image + gauss

    if noise_type == Noise.uniform:
        k = 0.5
        if image.ndim == 2:
            row, col = image.shape
            noise = np.random.rand(row, col)
        else:
            row, col, chan = image.shape
            noise = np.random.rand(row, col, chan)

        noise = image + k * (noise - 0.5)

    if noise_type == Noise.s_n_p:
        sh = image.shape
        s_vs_p = 0.5
        amount = 0.05
        noise = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in sh]
        noise[coords] = 1

        num_peper = np.ceil(amount * image.size * (1.0 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_peper)) for i in sh]
        noise[coords] = 0

    if noise_type == Noise.poisson:
        PEAK = 20
        noise = image + np.random.poisson(0.5 * PEAK, image.shape) / PEAK

    if noise_type == Noise.speckle:
        if image.ndim == 2:
            row, col = image.shape
            noise = np.random.randn(row, col)
        else:
            row, col, chan = image.shape
            noise = np.random.randn(row, col, chan)

        noise = image + image * 0.2 * noise
    noise[noise > 1] = 1
    noise[noise < 0] = 0
    return noise


def lowpass_filter(filter_type, image):
    if filter_type == Filter.mean_bilateral:
        image = img_as_ubyte(image)
        if image.ndim > 2:
            for i in range(image.ndim):
                image[:, :, i] = mean_bilateral(image[:, :, i], footprint=disk(20), s0=50, s1=180)
        else:
            image = mean_bilateral(image, footprint=disk(20), s0=50, s1=180)
        return image

    if filter_type == Filter.gauss:
        return gaussian(image, sigma=2, mode="reflect")

    if filter_type == Filter.mean:
        image = img_as_ubyte(image)
        footprint = disk(5)
        if image.ndim > 2:
            for i in range(image.ndim):
                image[:, :, i] = mean(image[:, :, i], footprint=footprint)
        else:
            image = mean(image, footprint=footprint)
        return image

    if filter_type == Filter.median:
        return median(image)


def read_images_3(input):

    images = ["motorcycle_left.png", "parrot.png", "cameraman.bmp", "horse.png"]

    row, column = 4, 2
    for noise in Noise:
        for i in range(len(images)):
            fig = plt.figure(images[i] + noise.value)
            plt.axis("off")
            img = io.imread(os.path.join(input, images[i]))
            img = img_as_float(img)
            i = 1

            fig.add_subplot(row, column, i, title=f"Original")
            plt.imshow(img)
            plt.axis("off")

            i += 1
            fig.add_subplot(row, column, i, title=f"{noise.value} Noise")
            img_noise = add_noise(noise, img)
            plt.imshow(img_noise)
            plt.axis("off")

            i += 1
            for _filter in Filter:
                fig.add_subplot(row, column, i, title=_filter.value)
                plt.imshow(lowpass_filter(_filter, img_noise.copy()))
                plt.axis("off")
                i += 1

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input1")
    read_images_3(dir_path)
