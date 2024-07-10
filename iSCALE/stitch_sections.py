import argparse

import numpy as np
import pandas as pd

from utils import (
        load_tsv, save_tsv, load_image, save_image,
        read_lines, load_pickle, save_pickle)
from stitch_images import stitch_images


def stitch_embeddings(embs_list, grid_shape, prefix):
    n_channels = len(embs_list[0])
    embs_new = []
    for i in range(n_channels):
        em_list = [embs[i] for embs in embs_list]
        em_new, __, __ = stitch_images(
                em_list, axis=0, grid_shape=grid_shape)
        if prefix is not None:
            save_pickle(em_new, f'{prefix}{i}.pickle')
            del em_new
        else:
            embs_new.append(em_new)
    if len(embs_new) == 0:
        embs_new = None
    return embs_new


def save_extent(origins, shapes, names=None, filename=None):
    origins, shapes = origins[:, ::-1], shapes[:, ::-1]  # ij to xy
    extents = np.concatenate([origins, shapes], axis=1)
    extents = pd.DataFrame(extents)
    extents.columns = ['origin_x', 'origin_y', 'length_x', 'length_y']
    if names is not None:
        extents.index = names
        extents.index.name = 'section'
    if filename is not None:
        save_tsv(extents, filename)
    return extents


def shift_coords(locs_list, origin_list):
    for locs, origin in zip(locs_list, origin_list):
        locs['y'] += origin[0]
        locs['x'] += origin[1]


def concat(df_list, name_list=None):
    if name_list is None:
        name_list = [f'{i:02d}_' for i in range(len(df_list))]
    for na, df in zip(name_list, df_list):
        df.index = na + '_' + df.index.astype(str)
    df = pd.concat(df_list, axis=0, join='inner')
    return df


def get_inputs(filename):
    lines = read_lines(filename)
    lists = [lin.split(' ') for lin in lines]
    names = [lis[0] for lis in lists]
    pref = [lis[1] for lis in lists]
    return names, pref


def intersect_cols(df_list):
    gene_names = set.intersection(*map(set, [df.columns for df in df_list]))
    gene_names = list(gene_names)
    return [df[gene_names] for df in df_list]


def union_cols(df_list, missing_val):
    gene_names = set.union(*map(set, [df.columns for df in df_list]))
    gene_names = list(gene_names)
    out_list = []
    for df in df_list:
        missing_names = [
                name for name in gene_names
                if name not in df.columns]
        missing_shape = (df.shape[0], len(missing_names))
        missing_dtype = df.dtypes[0]
        missing_df = pd.DataFrame(np.full(
            missing_shape, missing_val, dtype=missing_dtype))
        missing_df.columns = missing_names
        missing_df.index = df.index
        df = pd.concat([df, missing_df], axis=1)
        df = df[gene_names]
        out_list.append(df)
    return out_list


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inpfile', type=str)
    parser.add_argument('outpref', type=str)
    parser.add_argument('--grid', type=str, default=None)
    parser.add_argument('--balance-counts', action='store_true')
    parser.add_argument('--union-genes', action='store_true')
    parser.add_argument('--stitch-imgs', action='store_true')
    parser.add_argument('--stitch-locs', action='store_true')
    parser.add_argument('--stitch-cnts', action='store_true')
    parser.add_argument('--stitch-embs', action='store_true')
    parser.add_argument('--save-tiff', action='store_true')
    args = parser.parse_args()
    return args


def balance_counts(df_list):
    df_sum_original = pd.concat(df_list, axis=0).sum(0)
    df_list = [df - df.min() for df in df_list]
    df_list = [df / (df.max() + 1e-12) for df in df_list]
    df_sum = pd.concat(df_list, axis=0).sum(0)
    factor = df_sum_original / (df_sum + 1e-12)
    df_list = [df * factor for df in df_list]
    return df_list


def main():

    args = get_args()

    grid_shape = None
    if args.grid is not None:
        grid_shape = [int(s) for s in args.grid.split('x')]

    names, inppref_list = get_inputs(args.inpfile)

    if args.stitch_imgs:
        # stitch images
        img_list = [load_image(pref+'he.jpg') for pref in inppref_list]
        img, origins, shapes = stitch_images(
                img_list, axis=0, grid_shape=grid_shape)
        if args.save_tiff:
            extension = '.tif'
        else:
            extension = '.jpg'
        save_image(img, f'{args.outpref}he{extension}')
        save_extent(origins, shapes, names, args.outpref+'extents.tsv')

    # stitch locs
    if args.stitch_locs:
        locs_list = [load_tsv(pref+'locs.tsv') for pref in inppref_list]
        shift_coords(locs_list, origins)
        locs = concat(locs_list, names)
        save_tsv(locs, args.outpref+'locs.tsv')

    # stitch cnts
    if args.stitch_cnts:
        cnts_list = [load_tsv(pref+'cnts.tsv') for pref in inppref_list]

        if args.union_genes:
            cnts_list = union_cols(cnts_list, missing_val=0)
        else:
            cnts_list = intersect_cols(cnts_list)

        if args.balance_counts:
            cnts_list = balance_counts(cnts_list)

        cnts = concat(cnts_list, names)
        # sort by column variance
        cnts = cnts.loc[:, cnts.var().sort_values(ascending=False).index]
        save_tsv(cnts, args.outpref+'cnts.tsv')

    if args.stitch_embs:
        embs_list = [
                load_pickle(pref+'embeddings-hist.pickle')
                for pref in inppref_list]
        embs = {
                ke: stitch_embeddings(
                    [embs[ke] for embs in embs_list],
                    grid_shape=grid_shape,
                    prefix=f'{args.outpref}embeddings-hist/{ke}/')
                for ke in embs_list[0].keys()}
        save_pickle(embs, args.outpref+'embeddings-hist.pickle')


if __name__ == '__main__':
    main()
