import os.path
import pathlib
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.image import imsave, imread
from skimage import io, img_as_float32


def read_images_1(dir_path, imgs):

    for _img in imgs:

        # Original picture
        plt.figure(_img)
        plt.axis("off")
        img = imread(os.path.join(dir_path, _img))
        print(_img, img.ndim)
        if img.ndim > 2:
            img = img[:, :, 0]
        cmap = None
        plt.imshow(img, cmap=cmap)

        # Flip colors in picture
        plt.figure()
        plt.axis("off")
        flipped_cmap = cmap
        plt.imshow(img, cmap=flipped_cmap)

        # # Save picture
        # output_dir = os.path.join("images", "output")
        #
        # if not os.path.exists(output_dir):
        #     os.mkdir(output_dir)  # Create new folder if it does not exist.
        #
        # file_name, file_format = _img.split(".")
        # output_file_name = ".".join((os.path.join(output_dir, file_name + "2"), file_format))
        # try:
        #     imsave(output_file_name, img, cmap=flipped_cmap)
        # except KeyError as e:
        #     print(
        #         f"Unfortunately we cannot save file in {e} format via matplotlib.image.imsave.",
        #         f"Saving via tifffile.imwrite",
        #     )
        #     import tifffile
        #
        #     tifffile.imwrite(output_file_name, img)

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input1")
    read_images_1(dir_path, os.listdir(dir_path))
