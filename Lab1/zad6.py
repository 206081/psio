import os.path

import matplotlib.pyplot as plt
import pydicom
from skimage.exposure import equalize_adapthist
from skimage.util import img_as_ubyte


def contrast_equalize(img):
    img = equalize_adapthist(img, clip_limit=0.03)
    return img_as_ubyte(img)


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

        fig = plt.figure(_img)
        rows = 4
        columns = 2

        # Show original photo
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")

        # Show contrasted photo
        fig.add_subplot(rows, columns, 2)
        img2 = contrast_equalize(img)
        plt.imshow(img2, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.axis("off")

        for i, bins in enumerate([32, 64, 256]):
            # Histogram of original image with different bins.
            fig.add_subplot(rows, columns, 3 + i * 2, title=f"Bins: {bins}")
            plt.hist(img.ravel(), bins=bins)

            # Histogram of contrasted image with different bins.
            fig.add_subplot(rows, columns, 4 + i * 2, title=f"Equalized Bins: {bins}")
            plt.hist(img2.ravel(), bins=bins)

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = r"/input2"
    listdir = (file for file in os.listdir(dir_path) if file.split(".")[-1].upper() == "DCM")
    read_images_5(dir_path, listdir)
