import os
import pathlib

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from skimage import io
from skimage.segmentation import active_contour


def read_cat(_img):
    img = io.imread(_img)

    s = np.linspace(0, 2 * np.pi, 400)
    x = 110 + 40 * np.cos(s)
    y = 140 + 40 * np.sin(s)
    init1 = np.array([y, x]).T

    s = np.linspace(0, 2 * np.pi, 400)
    x = 470 + 320 * np.cos(s)
    y = 150 + 150 * np.sin(s)
    init2 = np.array([y, x]).T

    alpha = 0.0055
    beta = 25
    gamma = 0.0065
    # snake1 = active_contour(img, init1, alpha=alpha, beta=beta, gamma=gamma, max_iterations=2500, convergence=0.05)

    alpha = 0.37
    beta = 40
    gamma = 0.095

    snake2 = active_contour(img, init2, alpha=alpha, beta=beta, gamma=gamma, max_iterations=2500, convergence=0.001)
    """
    0.00652
    25
    0.00557
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 4), constrained_layout=True)
    axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
    alpha = Slider(
        ax=axfreq,
        label="Alpha",
        valmin=0.001,
        valmax=5,
        valinit=alpha,
    )
    axfreq = plt.axes([0.25, 0.2, 0.65, 0.03])
    beta = Slider(
        ax=axfreq,
        label="Beta",
        valmin=1,
        valmax=40,
        valinit=beta,
    )
    axfreq = plt.axes([0.25, 0.3, 0.65, 0.03])
    gamma = Slider(
        ax=axfreq,
        label="Gamma",
        valmin=0.001,
        valmax=0.1,
        valinit=gamma,
    )

    ax.imshow(img, cmap="gray")
    ax.set_title("")
    # ax.plot(init1[:, 1], init1[:, 0], "--r", lw=2)
    # (line,) = ax.plot(snake1[:, 1], snake1[:, 0], "--b", lw=2)
    ax.plot(init2[:, 1], init2[:, 0], "--r", lw=2)
    (line,) = ax.plot(snake2[:, 1], snake2[:, 0], "--b", lw=2)
    def update(val):


        snake = active_contour(
            img, init2, alpha=alpha.val, beta=beta.val, gamma=gamma.val, max_iterations=2500, convergence=0.05
        )
        line.set_ydata(snake[:, 0])
        line.set_xdata(snake[:, 1])
        fig.canvas.draw_idle()
        print("Done", val)


    alpha.on_changed(update)
    beta.on_changed(update)
    gamma.on_changed(update)

    plt.show(block=True)


if __name__ == "__main__":
    dir_path = os.path.join(pathlib.Path(__file__).parent.parent, "input4", "cat.jpg")
    read_cat(dir_path)
