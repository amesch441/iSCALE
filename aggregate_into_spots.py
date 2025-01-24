import sys
import os
from PIL import Image

import numpy as np
from einops import reduce, rearrange
import pandas as pd
from scipy.ndimage import label as label_connected
import matplotlib.pyplot as plt

from utils import (
        load_tsv, save_tsv, load_pickle, save_pickle,
        read_lines, write_lines, load_image)
from utils import save_image as save_img
from image import get_disk_mask
from visual import plot_matrix


def save_image(img, outfile):
    img = img.astype(np.float32)
    img -= np.nanmin(img)
    img /= np.nanmax(img) + 1e-12
    cmap = plt.get_cmap('turbo')
    img = cmap(img)[..., :3]
    save_img((img * 255).astype(np.uint8), outfile)


def get_data(prefix):
    cnts = load_tsv(prefix+'cnts-truth/cnts.tsv')
    gene_names = read_lines(prefix+'gene-names.txt')
    cnts = cnts[gene_names]
    locs = load_tsv(prefix+'cnts-truth/locs.tsv')
    assert (cnts.index == locs.index).all()
    mask = load_image(prefix+'cnts-truth/mask.png')
    mask = mask > 0
    if mask.ndim == 3:
        mask = mask.any(2)
    shape = Image.open(prefix+'he.jpg').size[::-1]
    shape = np.array(shape)
    return cnts, locs, mask, shape


def pad_mask(mask, shape):
    mask_new = np.zeros(shape, dtype=bool)
    mask_new[:mask.shape[0], :mask.shape[1]] = mask
    return mask_new


def fill_holes(mask):
    neg = ~mask
    labs = label_connected(neg)[0]
    lab_bg = labs[0, 0]
    is_bg = labs == lab_bg
    is_fg = ~is_bg
    return is_fg


def get_mask_spots(locs, shape, radius):
    locs = locs.to_numpy()
    mask = np.zeros(shape, dtype=bool)
    mask[locs[:, 0], locs[:, 1]] = True
    mask = reduce(
            mask, '(h0 h1) (w0 w1) -> h0 w0', 'max',
            h1=radius*2, w1=radius*2)
    mask = fill_holes(mask)
    return mask


def get_locs_arr(mask, radius):
    locs = np.meshgrid(
            np.arange(mask.shape[0]), np.arange(mask.shape[1]),
            indexing='ij')
    locs = np.stack(locs, -1)
    locs = locs.astype(np.float32)
    locs[~mask] = np.nan
    locs = reduce(
            locs, '(h0 h1) (w0 w1) k -> h0 w0 k', 'mean',
            h1=radius*2, w1=radius*2)
    return locs


def get_cnts_arr(cnts, locs, mask, radius, use_disk=False):
    cnts = cnts.to_numpy()
    locs = locs.to_numpy()
    if use_disk:
        disk = get_disk_mask(radius)
        print('Using circle spot masks...')
    else:
        print('Using square spot masks...')
    arr = []
    for i in range(cnts.shape[1]):
        print(i, '/', cnts.shape[1])
        ct = cnts[:, i]
        mat = np.full(mask.shape, np.nan)
        mat[mask] = 0
        mat[locs[:, 0], locs[:, 1]] = ct
        mat = rearrange(
                mat, '(h0 h1) (w0 w1) -> h0 w0 h1 w1',
                h1=radius*2, w1=radius*2)
        if use_disk:
            mat[..., ~disk] = 0
        mat = mat.sum((-1, -2))
        arr.append(mat)
    arr = np.stack(arr, -1)
    return arr


def to_df(cnts_arr, locs_arr, stride, gene_names):

    mask_cnts = np.isfinite(cnts_arr).all(-1)
    mask_locs = np.isfinite(locs_arr).all(-1)
    mask = np.logical_and(mask_cnts, mask_locs)
    mask = sieve_mask(mask, stride)

    cnts = cnts_arr[mask]
    locs = locs_arr[mask]

    spot_names = np.arange(cnts.shape[0])

    cnts = pd.DataFrame(cnts)
    cnts.columns = gene_names
    cnts.index = spot_names
    cnts.index.name = 'spot'

    locs = np.round(locs).astype(int)
    locs = locs[:, ::-1]  # ij to xy
    locs = pd.DataFrame(locs)
    locs.columns = ['x', 'y']
    locs.index.name = 'spot'

    return cnts, locs


def sieve_mask(mask, stride):
    filt = np.zeros_like(mask)
    filt[::stride, ::stride] = True
    mask = np.logical_and(mask, filt)
    return mask


def plot_arr(x, names, prefix):
    for i, nam in enumerate(names):
        outfile = prefix + nam + '.png'
        plot_matrix(x[..., i], outfile)


def main():

    prefix = sys.argv[1]
    radius = int(sys.argv[2])  # e.g. 64
    stride = int(sys.argv[3])  # e.g. 2
    geometry = sys.argv[4]  # square or circle

    cnts, locs, mask, shape = get_data(prefix)
    mask = pad_mask(mask, shape)
    locs = locs[['y', 'x']]  # xy to ij
    # mask = get_mask_spots(locs, shape, radius)


    print(cnts)
    print(locs)
    print(mask.shape)
    print(cnts.shape)
    print(locs.shape)

    prefix_out = (
            f'{prefix}cnts-truth-agg/'
            f'radius{radius:04d}-stride{stride:02d}-{geometry}/')
    cache_file = prefix_out + 'data.pickle'

    if os.path.exists(cache_file):
        data = load_pickle(cache_file)
        cnts_arr, locs_arr = data['cnts'], data['locs']
        gene_names = data['gene_names']
    else:
        locs_arr = get_locs_arr(mask, radius)
        use_disk_dict = {'square': False, 'circle': True}
        use_disk = use_disk_dict[geometry]
        cnts_arr = get_cnts_arr(
                cnts=cnts, locs=locs, mask=mask, radius=radius,
                use_disk=use_disk)
        gene_names = cnts.columns.to_list()
        save_pickle(
            dict(
                cnts=cnts_arr,
                locs=locs_arr,
                gene_names=gene_names),
            cache_file)

    write_lines([radius], prefix_out+'radius.txt')
    cnts_df, locs_df = to_df(
            cnts_arr=cnts_arr, locs_arr=locs_arr,
            stride=stride, gene_names=gene_names)
    save_tsv(locs_df, prefix_out+'locs.tsv')
    save_tsv(cnts_df, prefix_out+'cnts.tsv')
    plot_arr(cnts_arr, gene_names, prefix_out+'plots/')


if __name__ == '__main__':
    main()
