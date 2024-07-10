import sys

import numpy as np

from utils import load_pickle, save_pickle
from image import crop_image


def stitch_embeddings(embs_list):
    shape_list = [embs[0].shape for embs in embs_list]
    shape = np.max(shape_list, axis=0)
    extent = [(0, s) for s in shape]
    n_channels_list = [len(embs) for embs in embs_list]
    assert np.unique(n_channels_list).size == 1
    n_channels = n_channels_list[0]
    channels = []
    for i in range(n_channels):
        x_secs = [crop_image(embs[i], extent) for embs in embs_list]
        x = np.concatenate(x_secs, axis=1)
        channels.append(x)
    return channels


def main():
    outfile = sys.argv[1]
    inpfile_list = sys.argv[2:]

    embs_list = [load_pickle(fname) for fname in inpfile_list]
    embs = {
            ke: stitch_embeddings([embs[ke] for embs in embs_list])
            for ke in embs_list[0].keys()}

    save_pickle(embs, outfile)


if __name__ == '__main__':
    main()
