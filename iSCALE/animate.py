import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils import load_pickle
from visual import plot_labels_3d


def update(i, fig, ax):
    ax.view_init(elev=20., azim=i)
    return fig, ax


def get_data():

    labels = load_pickle('tmp/labels.pickle')
    labels = labels.transpose(1, 2, 0)  # row, column, floor
    labels = labels[::-1, :, ::-1]  # adjust for 3d plotting orientation

    labs_to_show = {
            'immune': [8],
            'insitu': [7, 9],
            'invasive': [2, 4, 5, 6],
            }

    return labels, labs_to_show


def get_colors(labels):
    colors = plot_labels_3d(labels)
    colors = (colors / 255.0)
    colors_alpha = 0.5  # opacity
    colors = np.concatenate(
            [colors, np.full_like(colors[..., [0]], colors_alpha)], -1)
    return colors


def animate(mask, colors, filename):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    y, x, z = np.where(mask)
    ax.scatter(x, y, z, c=colors, s=0.01)

    plt.axis('off')
    fig.set_size_inches(5, 5)

    anim = FuncAnimation(
            fig, update, frames=np.arange(0, 360, 2),
            repeat=True, fargs=(fig, ax))
    anim.save(filename, dpi=100, writer='imagemagick', fps=24)
    print(filename)


labels, labs_to_show_dict = get_data()
colors = get_colors(labels)
for labs_name, labs_to_show in labs_to_show_dict.items():
    outfile = f'tmp/{labs_name}.gif'
    mask = np.isin(labels, labs_to_show)
    print(labs_name, mask.mean())
    co = colors[mask]
    animate(mask, co, outfile)
