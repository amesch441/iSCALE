import sys

import numpy as np
import pandas as pd

from utils import save_tsv, read_string, load_pickle
from impute import get_data, get_patches_flat
from image import get_disk_mask
from evaluate_imputed import metric_fin


def evaluate_fit(y, y_fit, gene_names, filename):
    eval = {
            met: np.array([
                metric_fin(c, cf, met)
                for c, cf in zip(y.T, y_fit.T)])
            for met in ['rmse', 'pearson']}
    df = pd.DataFrame(eval)
    df.index = gene_names
    df.index.name = 'gene'
    df = df.round(4)
    save_tsv(df, filename)


def get_fit(prefix, gene_names, locs, radius):
    mask = get_disk_mask(radius)
    cnts_fit = []
    for gname in gene_names:

        c_pred = load_pickle(
                f'{prefix}cnts-super/{gname}.pickle', verbose=False)
        c_pred -= np.nanmin(c_pred)
        c_pred /= np.nanmax(c_pred) + 1e-12

        c_fit = get_patches_flat(c_pred, locs, mask)
        c_fit = c_fit.mean(-1)

        cnts_fit.append(c_fit)
    cnts_fit = np.array(cnts_fit).T
    return cnts_fit


def main():

    prefix = sys.argv[1]  # e.g. data/her2st/H123/

    __, cnts_obsr, locs = get_data(prefix)
    radius = int(read_string(f'{prefix}radius.txt'))
    factor = 16
    radius = np.round(radius / factor).astype(int)

    gene_names = cnts_obsr.columns
    cnts_fit = get_fit(prefix, gene_names, locs, radius)
    evaluate_fit(
            cnts_obsr.to_numpy(), cnts_fit, gene_names,
            f'{prefix}cnts-super-eval/factor9100.tsv')


if __name__ == '__main__':
    main()
