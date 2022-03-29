import os.path
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_float
from skimage.filters import gaussian
from skimage.filters.rank import mean, mean_bilateral, median
from skimage.morphology import disk, cube


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
        noise[noise > 1] = 1
        noise[noise < 0] = 0
        return noise

    if noise_type == Noise.uniform:
        k = 0.5
        if image.ndim == 2:
            row, col = image.shape
            noise = np.random.rand(row, col)
        else:
            row, col, chan = image.shape
            noise = np.random.rand(row, col, chan)

        noise = image + k * (noise - 0.5)

        return noise

    if noise_type == Noise.s_n_p:
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

    if noise_type == Noise.poisson:
        PEAK = 20
        noisy = image + np.random.poisson(0.5 * PEAK, image.shape) / PEAK
        return noisy

    if noise_type == Noise.speckle:
        if image.ndim == 2:
            row, col = image.shape
            noise = np.random.randn(row, col)
        else:
            row, col, chan = image.shape
            noise = np.random.randn(row, col, chan)

        noisy = image + image * 0.2 * noise
        return noisy


def lowpass_filter(filter_type, image):
    if filter_type == Filter.gauss:
        return gaussian(image, sigma=2, mode="reflect")

    if filter_type == Filter.mean:
        return mean(image, footprint=cube(20))

    if filter_type == Filter.mean_bilateral:
        print(image)
        a = [0] * 3
        n = np.zeros((2,3))
        for i in range(3):
            a[i] = mean_bilateral(image[:, :, i], footprint=disk(10), s0=50, s1=50)
        print("a", a)
        for x,y,z in a:
            for i,j,k in zip(x,y,z):
                n.put(i,j,k)
        print("n", n)

    if filter_type == Filter.median:
        return median(image)


def read_images_3(input):

    images = ["motorcycle_left.png", "parrot.png", "cameraman.bmp", "horse.png"]
    """
    s&p median
    gauss 
    uniform
    poisson
    speckle
    """
    row, column = 4, 2
    # Original picture
    fig = plt.figure("Zad3")
    plt.axis("off")
    moto = io.imread(os.path.join(input, images[0]))
    moto = img_as_float(moto)
    i = 1

    noise = Noise.gauss
    fig.add_subplot(row, column, i, title=f"Original")
    plt.imshow(moto)
    plt.axis("off")

    i += 1
    fig.add_subplot(row, column, i, title=f"{noise.value} Noise")
    moto_noise = add_noise(noise, moto)
    plt.imshow(moto_noise)
    plt.axis("off")

    i += 1
    for filter in Filter:
        print(filter)
        fig.add_subplot(row, column, i, title=filter.value)
        plt.imshow(lowpass_filter(filter, moto_noise))
        plt.axis("off")
        i += 1

    # fig.add_subplot(row, column, i, title="Uniform Noise")
    # uniform = add_noise("uniform", img)
    # plt.imshow(uniform, cmap="gray")
    # plt.axis("off")
    # i += 1
    #
    # fig.add_subplot(row, column, i, title="Filtered Mean")
    # plt.imshow(lowpass_filter("mean", uniform), cmap="gray")
    # plt.axis("off")
    # i += 1
    #
    # fig.add_subplot(row, column, i, title="S&P Noise")
    # s_and_p = add_noise("s&p", img)
    # plt.imshow(s_and_p, cmap="gray")
    # plt.axis("off")
    # i += 1
    #
    # fig.add_subplot(row, column, i, title="Filtered Median")
    # plt.imshow(lowpass_filter("median", s_and_p), cmap="gray")
    # plt.axis("off")
    # i += 1
    #
    # fig.add_subplot(row, column, i, title="Speckle Noise")
    # speckle = add_noise("speckle", data.coins())
    # plt.imshow(speckle, cmap="gray")
    # plt.axis("off")
    # i += 1
    #
    # fig.add_subplot(row, column, i, title="Filtered Mean Bilateral")
    # plt.imshow(lowpass_filter("mean_bilateral", speckle), cmap="gray")
    # plt.axis("off")

    plt.show(block=True)


if __name__ == "__main__":
    read_images_3(r"/home/michal/PycharmProjects/pythonProject/input1")
