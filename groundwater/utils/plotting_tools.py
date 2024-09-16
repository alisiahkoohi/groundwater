# # Adapted from https://github.com/devitocodes/devito/blob/master/examples/cfd/09_Darcy_flow_equation.ipynb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


font = {"family": "serif", "style": "normal", "size": 12}
matplotlib.rc("font", **font)


def plot_fields(fields, titles, suptitle, contour=False):
    fig, axes = plt.subplots(1, len(fields), figsize=(5 * len(fields), 5))
    if len(fields) == 1:
        axes = [axes]
    for ax, field, title in zip(axes, fields, titles):
        im = ax.imshow(
            np.exp(field),
            interpolation="lanczos",
            origin="lower",
            cmap="Blues_r",
        )
        ax.set_title(title)

        # Add contour plot if contour=True
        if contour:
            cs = ax.contour(
                np.exp(field),
                colors="red",
                alpha=1.0,
                linewidths=1,
            )
            # ax.clabel(cs, inline=True, fontsize=8)  # Add labels to contours

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax.grid(False)

    fig.suptitle(suptitle)
    plt.tight_layout()
