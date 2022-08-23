import os.path
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pydicom
from skimage import img_as_ubyte
from skimage.exposure import rescale_intensity


def contrast_stretch_3D(img):
    v_min, v_max = np.percentile(img, (2, 98.0))
    return rescale_intensity(img, in_range=(v_min, v_max))


def read_images_5(dir_path, imgs):
    for _img in imgs:

        # Read image
        img = pydicom.dcmread(os.path.join(dir_path, _img))

        img = img.pixel_array
        if img.ndim < 3:
            print(f"Image {_img} is not 3D.")
            continue
        img = img_as_ubyte(img[0])

        vmin = img.min()
        vmax = img.max()
        cmap = plt.cm.bone

        # Show original image
        plt.figure()
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")

        # Show contrasted image
        plt.figure()
        img2 = contrast_stretch_3D(img)
        plt.imshow(img2, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")

        # Show histograms
        fig = plt.figure("Hist" + _img)
        rows = 1
        columns = 2

        # of original image
        fig.add_subplot(rows, columns, 1)
        plt.hist(img.ravel(), bins=50)

        # of contrasted image
        fig.add_subplot(rows, columns, 2)
        plt.hist(img2.ravel(), bins=50)

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input2")
    listdir = (file for file in os.listdir(dir_path) if file.split(".")[-1].upper() == "DCM")
    read_images_5(dir_path, listdir)
