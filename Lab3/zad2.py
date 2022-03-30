import os
import pathlib

from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage import io, img_as_float64
from skimage.feature import canny
from skimage.filters.edges import prewitt, sobel
from skimage.filters.thresholding import apply_hysteresis_threshold
from skimage.morphology import dilation, disk, erosion


def read_gears(_img):

    # Read image
    _img = io.imread(_img, as_gray=True)
    row = 1
    column = 3
    _img_as_float = 1 - img_as_float64(_img)

    # ------------------------------------------------------------------------------------------------------------------
    plt.figure("Filling Holes")
    plt.axis("off")
    plt.subplot(row, column, 1, title="Original →")
    plt.imshow(_img_as_float, cmap="gray")

    hysteresis = apply_hysteresis_threshold(_img_as_float, 0.25, 0.5)
    plt.subplot(row, column, 2, title="Hysteresis →")
    plt.imshow(hysteresis, cmap="gray")

    binary_hysteresis = binary_fill_holes(hysteresis)
    plt.subplot(row, column, 3, title="Binary Fill Holes")
    plt.imshow(binary_hysteresis, cmap="gray")

    # ------------------------------------------------------------------------------------------------------------------
    plt.figure("Canny")
    plt.axis("off")
    plt.subplot(row, column + 1, 1, title="Original →")
    plt.imshow(_img, cmap="gray")

    _canny = canny(1 - _img, sigma=1.5, low_threshold=0.15, high_threshold=40)
    plt.subplot(row, column + 1, 2, title="Canny →")
    plt.imshow(_canny, cmap="gray")

    _dilation = dilation(_canny)
    plt.subplot(row, column + 1, 3, title="Dilation →")
    plt.imshow(_dilation, cmap="gray")

    binary_canny = binary_fill_holes(_dilation)
    plt.subplot(row, column + 1, 4, title="Binary Fill Holes")
    plt.imshow(binary_canny, cmap="gray")

    # ------------------------------------------------------------------------------------------------------------------
    plt.figure("Prewitt")
    plt.axis("off")
    plt.subplot(row, column + 2, 1, title="Original →")
    plt.imshow(_img, cmap="gray")

    _prewitt = prewitt(_img_as_float)
    plt.subplot(row, column + 2, 2, title="Prewitt →")
    plt.imshow(_prewitt, cmap="gray")

    _erosion_prewitt = erosion(_prewitt, footprint=disk(1))
    plt.subplot(row, column + 2, 3, title="Erosion →")
    plt.imshow(_erosion_prewitt, cmap="gray")

    _dilation_prewitt = dilation(_erosion_prewitt, footprint=disk(1))
    plt.subplot(row, column + 2, 4, title="Dilation →")
    plt.imshow(_dilation_prewitt, cmap="gray")

    binary_prewitt = binary_fill_holes(_dilation_prewitt)
    plt.subplot(row, column + 2, 5, title="Binary Fill Holes")
    plt.imshow(binary_prewitt, cmap="gray")

    # ------------------------------------------------------------------------------------------------------------------
    plt.figure("Sobel")
    plt.axis("off")
    plt.subplot(row, column, 1, title="Original →")
    plt.imshow(_img, cmap="gray")

    _sobel = sobel(_img_as_float)
    plt.subplot(row, column, 2, title="Sobel →")
    plt.imshow(_sobel, cmap="gray")

    binary_sobel = binary_fill_holes(_sobel)
    plt.subplot(row, column, 3, title="Binary Fill Holes")
    plt.imshow(binary_sobel, cmap="gray")

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input3", "gears.bmp")
    read_gears(dir_path)
