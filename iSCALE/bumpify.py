from time import time
import sys

import numpy as np
# from sklearn.neighbors import KNeighborsRegressor
from cuml.neighbors import KNeighborsRegressor
from einops import reduce

from utils import load_pickle, save_pickle
from aggregate_into_spots import save_image
from image import upscale, smoothen


def smooth(x, factor):
    shape = x.shape[:2]
    x = reduce(
            x, '(h1 h) (w1 w) c -> h1 w1 c', 'mean', h=factor, w=factor)
    x = upscale(x, shape)
    return x


def neighbor_regress(x, y):
    print(x.shape, y.shape)
    t0 = time()
    model = KNeighborsRegressor(n_neighbors=2, weights='uniform')
    model.fit(x, y)
    y_new = model.predict(x)
    return y_new
    print(int(time() - t0), 'sec')


def bumpify_global(x, y):
    mask = np.isfinite(x[..., 0])
    y_new = np.full_like(y, np.nan)
    y_new_flat = neighbor_regress(x[mask], y[mask])
    y_new[mask] = y_new_flat
    return y_new


def bumpify_local(x, y, labels):
    labs_uniq = np.unique(labels)
    labs_uniq = labs_uniq[labs_uniq >= 0]
    y_new = np.full_like(y, np.nan)
    for lab in labs_uniq:
        mask = labels == lab
        y_new_flat = neighbor_regress(x[mask], y[mask])
        y_new[mask] = y_new_flat
    return y_new


def visualize(y, y_new, gene_names, prefix):
    gene_names = np.array(gene_names)
    gene_names_short = ['ERBB2', 'CD74']
    for gname in gene_names_short:
        idx = np.where(gene_names == gname)[0][0]
        save_image(y[..., idx], f'{prefix}ori/{gname}.png')
        save_image(y_new[..., idx], f'{prefix}new/{gname}.png')


def main():

    prefix = sys.argv[1]  # data/stable/her2st/G1/
    method = 'global'

    # gene_names = read_lines(prefix+'gene-names.txt')
    gene_names = ['ERBB2', 'CD74']
    cnts = [
            load_pickle(f'{prefix}cnts-super/{gname}.pickle')
            for gname in gene_names]
    embs = load_pickle(prefix+'embeddings-hist.pickle')
    labels = load_pickle(prefix+'clusters-gene/labels.pickle')

    x = np.stack(embs['sub'], -1)
    y = np.stack(cnts, -1)

    if method == 'global':
        y_raw = bumpify_global(x, y)
    elif method == 'local':
        y_raw = bumpify_local(x, y, labels)
    elif method == 'load':
        y_raw = load_pickle('tmp/bumpify/new/raw.pickle')
    else:
        raise ValueError('Method not recognized')

    save_pickle(y_raw, 'tmp/bumpify/new/raw.pickle')

    y_res = y_raw - y
    y_res_smooth = smoothen(y_res, size=8)
    # y_res_smooth = smooth(y_res, factor=16)
    y_bum = y_raw - y_res_smooth

    visualize(y, y_bum, gene_names, 'tmp/bumpify/')


if __name__ == '__main__':
    main()
