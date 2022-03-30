import os
import pathlib

from matplotlib import pyplot as plt
from skimage import io, img_as_float
from skimage.color import rgb2gray
from skimage.filters.edges import sobel
from skimage.segmentation import felzenszwalb, watershed, mark_boundaries, slic, quickshift


def read_boundaries(_img):

    # Read image
    _img = io.imread(_img)
    row = 3
    column = 2
    _img_as_float = img_as_float(_img[::2, ::2])

    # ------------------------------------------------------------------------------------------------------------------
    plt.figure("Segmentation")
    plt.axis("off")
    plt.subplot(row, column, 1, title="Original")
    plt.imshow(_img_as_float)

    # ---------------------------------------------------------------------------------------------------------------- #
    segments_fz = felzenszwalb(_img_as_float, scale=100, sigma=0.5, min_size=50)
    plt.subplot(row, column, 3, title="Felzenszwalb")
    plt.imshow(mark_boundaries(_img_as_float, segments_fz))

    # ---------------------------------------------------------------------------------------------------------------- #
    gradient = sobel(rgb2gray(_img_as_float))
    segments_watershed = watershed(gradient, markers=250, compactness=0.001)
    plt.subplot(row, column, 4, title="Watershed")
    plt.imshow(mark_boundaries(_img_as_float, segments_watershed))

    # ---------------------------------------------------------------------------------------------------------------- #
    _slic = slic(_img_as_float, n_segments=250, compactness=10, sigma=1, start_label=1)
    plt.subplot(row, column, 5, title="SLIC")
    plt.imshow(mark_boundaries(_img_as_float, _slic))

    # ---------------------------------------------------------------------------------------------------------------- #
    _quick = quickshift(_img_as_float, kernel_size=3, max_dist=6, ratio=0.5)
    plt.subplot(row, column, 6, title="Quick")
    plt.imshow(mark_boundaries(_img_as_float, _quick))

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input3", "fish.bmp")
    read_boundaries(dir_path)
