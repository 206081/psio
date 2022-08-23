import os.path
from pathlib import Path
import pathlib

import matplotlib.pyplot as plt
from skimage.io import imsave, imread


def read_images_2(dir_path, imgs):
    for _img in imgs:

        # Set figure
        fig = plt.figure()
        rows = 1
        columns = 2

        # Read image
        img = imread(os.path.join(dir_path, _img))

        # Add first image
        fig.add_subplot(rows, columns, 1)
        plt.imshow(img)
        plt.axis("off")

        # Add mirrored image
        fig.add_subplot(rows, columns, 2)
        img = img[:, ::-1]
        plt.imshow(img)
        plt.axis("off")

        # Save picture
        # output_dir = os.path.join("images", "output")
        #
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)  # Create new folder if it does not exist.
        #
        # file_name, file_format = _img.split(".")
        # output_file_name = ".".join((os.path.join(output_dir, file_name + "3"), file_format))
        # imsave(output_file_name, img)

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input1")
    read_images_2(dir_path, os.listdir(dir_path))
