import os.path
import pathlib
from pathlib import Path

from cv2 import imread, imshow, imwrite, cvtColor, COLOR_BGR2GRAY, waitKey


def read_images_3(dir_path, imgs):
    for _img in imgs:

        # Read image
        img_path = os.path.join(dir_path, _img)
        img = imread(img_path)
        if img is None:
            print(f"We cannot read a {_img} file via openCV")
            continue
        img_lumi = cvtColor(img, COLOR_BGR2GRAY)

        imshow(_img, img)

        # Add mirrored image
        img = img[:, ::-1]
        imshow("flip" + _img, img)
        imshow("8bit" + _img, img_lumi)

        # Save picture
        output_dir = os.path.join("images", "output")

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)  # Create new folder if it does not exist.

        file_name, file_format = _img.split(".")
        file_name = os.path.join(output_dir, file_name + "4")
        output_file_name = ".".join((file_name, file_format))
        output_file_name_lumi = ".".join((file_name + "lumi", file_format))
        imwrite(output_file_name, img)
        imwrite(output_file_name_lumi, img_lumi)

    waitKey(0)


if __name__ == "__main__":
    dir_path = os.path.join("images", "../input1")
    read_images_3(dir_path, os.listdir(dir_path))
