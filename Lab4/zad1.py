import os
import pathlib

from matplotlib import pyplot as plt
from numpy.ma import masked_where
from skimage import io
from skimage.segmentation import random_walker


def read_lungs(_img):

    _img_seed_1_path = _img.replace(".bmp", "_seeds1.bmp")
    _img_seed_2_path = _img.replace(".bmp", "_seeds2.bmp")
    _img_seed_3_path = _img.replace(".bmp", "_seeds3.bmp")

    _img = io.imread(_img, as_gray=True)
    _img_seed_1 = io.imread(_img_seed_1_path, as_gray=True)
    _img_seed_2 = io.imread(_img_seed_2_path, as_gray=True)
    _img_seed_3 = io.imread(_img_seed_3_path, as_gray=True)

    row = 1

    seeds = (
        _img_seed_1,
        # _img_seed_2,
        # _img_seed_3,
    )
    beta = (10000,)
    column = len(beta) * 2 + 2

    # for i, seed in enumerate(seeds, 1):
    #     plt.figure(i)
    #
    #     for j, _beta in enumerate(beta):
    #         plt.subplot(row, column, 2 + j * 2)
    #         plt.imshow(seed, cmap=plt.cm.bone)
    #         walker = random_walker(_img, seed, beta=_beta)
    #         plt.subplot(row, column, 3 + j * 2)
    #         plt.imshow(walker, cmap=plt.cm.bone)
    #
    #         cont = find_contours(walker)
    #         ax = plt.subplot(row, column, 1 + j * 2)
    #         ax.imshow(_img, cmap=plt.cm.bone)
    #
    #         for x, cnt in enumerate(cont):
    #             ax.plot(cnt[:, 1], cnt[:, 0], "-r", lw=2)

    for i, seed in enumerate(seeds, 1):
        plt.figure(i)

        for j, _beta in enumerate(beta):
            # plt.subplot(row, column, 2 + j * 2)
            # plt.imshow(seed, cmap=plt.cm.bone)
            walker = random_walker(_img, seed, beta=_beta)
            masked = masked_where(walker == 1, walker)
            ax = plt.subplot(row, column, 3 + j * 2)
            ax.imshow(masked, cmap=plt.cm.bone)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax = plt.subplot(row, column,   + j * 2)
            ax.imshow(_img, cmap=plt.cm.bone)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            # ax.imshow(_img, cmap=plt.cm.bone)
            # ax.imshow(masked, interpolation="none", cmap="autumn", alpha=1)
    # plt.axis("off")
    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input4", "lungs_lesion.bmp")
    read_lungs(dir_path)
