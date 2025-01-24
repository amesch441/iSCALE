import argparse

import numpy as np

from utils import load_pickle, load_mask
from reduce_dim import reduce_dim
from visual import plot_matrix


def plot_embeddings(embs, n_channels, prefix):
    for key, channels in embs.items():
        for i in range(min(n_channels, len(channels))):
            plot_matrix(
                    channels[i], f'{prefix}{key}/{i:03d}.png',
                    white_background=True)


def reduce_dim_transpose(x, **kwargs):
    x = reduce_dim(np.stack(x, -1), **kwargs)[0]
    x = x.transpose(2, 0, 1)
    return x


def reduce_dimension(embs, n_components):
    return {
            ke: reduce_dim_transpose(
                e, n_components=min(len(e), n_components), method='pca')
            for ke, e in embs.items()}


def remove_background(embs, mask):
    for ke, channels in embs.items():
        for chan in channels:
            chan[~mask] = np.nan


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('embeddings', type=str)
    parser.add_argument('prefix', type=str)
    parser.add_argument('--mask', type=str, default=None)
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    n_channels = 100

    embs = load_pickle(args.embeddings)

    if args.mask is not None:
        mask = load_mask(args.mask)
        remove_background(embs, mask)

    plot_embeddings(
            embs, n_channels=n_channels, prefix=args.prefix+'raw/')

    embs = reduce_dimension(embs, n_components=n_channels)
    plot_embeddings(
            embs, n_channels=n_channels, prefix=args.prefix+'dim-reduced/')


if __name__ == '__main__':
    main()
