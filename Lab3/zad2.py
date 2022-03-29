from cv2 import imread, threshold, THRESH_BINARY, resize, adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C
from matplotlib import pyplot as plt
from numpy import asarray
from scipy.ndimage import binary_fill_holes, prewitt
from skimage import img_as_ubyte, img_as_float32, io, img_as_float64
from skimage.feature import canny

from skimage.filters import thresholding
from skimage.filters.thresholding import apply_hysteresis_threshold
from skimage.morphology import remove_small_objects


def read_gears(_img):

    # Read image
    _img = io.imread(_img)
    row = 3
    column = 2
    _img = 1 - img_as_float64(_img)

    plt.subplot(row, column, 1)
    plt.imshow(_img, cmap="gray")

    img = apply_hysteresis_threshold(_img, 0.25, 0.5)
    plt.subplot(row, column, 2)
    plt.imshow(img, cmap="gray")

    img = canny(_img, sigma=1.5, low_threshold=0.15, high_threshold=0.4)
    plt.subplot(row, column, 3)
    plt.imshow(img, cmap="gray")

    img2 = binary_fill_holes(img)
    plt.subplot(row, column, 4)
    plt.imshow(img2, cmap="gray")

    img3 = prewitt(img)
    plt.subplot(row, column, 5)
    plt.imshow(img3, cmap="gray")


    plt.show(block=True)


if __name__ == "__main__":
    read_gears(r"/home/michal/PycharmProjects/pythonProject/gears.bmp")
