import sys

import numpy as np
import pandas as pd

from utils import load_pickle, save_tsv, write_lines, read_string
from image import get_disk_mask
from impute import get_patches_flat


def get_hexagonal_tiling(size, step, margin):
    offset = (step * 0.5 * 3**0.5, step * 0.5)
    a = np.arange(margin, size[1]-offset[1]-margin, step)
    a = np.stack([np.zeros_like(a), a], -1)
    b = a + [offset]
    ab = np.concatenate([a, b])
    anchors = np.arange(margin, size[0]-offset[0]-margin, offset[0]*2)
    anchors = np.stack([anchors, np.zeros_like(anchors)], -1)
    locs = np.expand_dims(ab, 0) + np.expand_dims(anchors, 1)
    locs = locs.reshape(-1, locs.shape[-1])
    return locs

    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(8, 8))
    # plt.plot(locs[:, 1], locs[:, 0], '.')
    # plt.gca().invert_yaxis()
    # plt.gca().set_aspect('equal')
    # plt.savefig('a.png', dpi=300, bbox_inches='tight')
    # plt.close()


def save_data(cnts, locs, radius, gene_names, prefix):

    locs = pd.DataFrame(locs)
    locs.columns = ['y', 'x']
    locs.index.name = 'spot'
    locs = locs.astype(int)
    locs = locs[['x', 'y']]
    save_tsv(locs, prefix+'locs.tsv')

    cnts = pd.DataFrame(cnts)
    cnts.columns = gene_names
    cnts.index.name = 'spot'
    save_tsv(cnts, prefix+'cnts.tsv')

    write_lines([radius], prefix+'radius.txt')


def main():

    prefix = sys.argv[1]  # e.g. 'data/xenium/rep1/'

    spot_size = 55  # diameter of spot in microns
    step_size = 100  # center-to-center distance in microns

    pixel_size = float(read_string(prefix+'pixel-size.txt'))
    radius = spot_size / pixel_size * 0.5
    step = step_size / pixel_size

    factor = 8
    data = load_pickle(
            f'{prefix}cnts-truth-agg/radius{factor//2:04d}-'
            'stride01-square/data.pickle')
    cnts_arr = data['cnts']
    gene_names = data['gene_names']



    step_scaled = np.round(step / factor).astype(int)
    radius_scaled = np.round(radius / factor).astype(int)

    locs = get_hexagonal_tiling(
            size=cnts_arr.shape[:2],
            step=step_scaled,
            margin=radius_scaled)
    locs = locs.astype(int)
    cnts = get_patches_flat(
            img=cnts_arr,
            locs=locs,
            mask=get_disk_mask(radius_scaled))
    cnts = cnts.sum(1)

    isfin = np.isfinite(cnts).all(1)
    cnts, locs = cnts[isfin], locs[isfin]
    locs *= factor

    save_data(
            cnts=cnts, locs=locs, radius=radius_scaled*factor,
            gene_names=gene_names,
            prefix=f'{prefix}cnts-truth-agg/hexagonal/')


if __name__ == '__main__':
    main()
