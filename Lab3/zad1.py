import os
import pathlib

from cv2 import threshold, THRESH_BINARY, adaptiveThreshold, ADAPTIVE_THRESH_MEAN_C
from matplotlib import pyplot as plt
from scipy.ndimage import binary_fill_holes
from skimage import img_as_ubyte, io, img_as_float64
from skimage.filters import thresholding
from skimage.morphology import remove_small_objects


def read_tumor(_img):

    # Read image
    _img = io.imread(_img)
    row = 3
    column = 2
    plt.subplot(row, column, 1)
    plt.imshow(_img)
    _, img = threshold(_img, 230, 255, THRESH_BINARY)
    plt.subplot(row, column, 2)
    plt.imshow(img)
    img = img_as_ubyte(_img)
    img = img > 230
    img = img.astype(bool)
    img = binary_fill_holes(img)
    img = remove_small_objects(img, min_size=400, connectivity=2)
    plt.subplot(row, column, 3)
    plt.imshow(img)

    img = img_as_float64(_img)
    thr = thresholding.threshold_otsu(img)
    img = img >= thr
    plt.subplot(row, column, 4)
    plt.imshow(img)

    img = img_as_ubyte(_img)
    thr = adaptiveThreshold(img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 11, 0)
    img = img >= thr
    plt.subplot(row, column, 5)
    plt.imshow(img)

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input3", "brain_tumor.bmp")
    read_tumor(dir_path)
