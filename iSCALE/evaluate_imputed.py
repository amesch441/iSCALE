import sys

import numpy as np
from scipy.stats import pearsonr, spearmanr
from einops import reduce
from cv2 import fastNlMeansDenoising
from structural_similarity import structural_similarity
import pandas as pd

from utils import load_pickle, save_tsv, load_mask
from visual import plot_matrix, plot_cells
from plot_cells import aggregate_by_weights


def corr_pearson_stablized(x, y, epsilon=1e-6):
    x = standardize(x)
    y = standardize(y)
    x = x - x.mean()
    y = y - y.mean()
    x_std = (x**2).mean()**0.5
    y_std = (y**2).mean()**0.5
    corr = ((x * y).mean() + epsilon) / (x_std * y_std + epsilon)
    return corr


def corr_pearson(x, y):
    return pearsonr(x, y)[0]


def corr_spearman(x, y):
    return spearmanr(x, y)[0]


def corr_uncentered(x, y):
    return np.mean(x * y) / np.mean(x**2)**0.5 / np.mean(y**2)**0.5


def normalize(x):
    x = x - np.nanmean(x)
    x = x / (np.nanstd(x) + 1e-12)
    return x


def standardize(x):
    x = x - np.nanmin(x)
    x = x / (np.nanmax(x) + 1e-12)
    return x


def psnr(x, y):
    cutoff = 4.0
    x = np.clip(normalize(x), -cutoff, cutoff)
    y = np.clip(normalize(y), -cutoff, cutoff)
    mse = np.square(x - y).mean()
    psnr = 10 * np.log10(cutoff**2 / mse)
    return psnr


def rectangular_hull(mask):
    indices = np.stack(np.where(mask), -1)
    start = indices.min(0)
    stop = indices.max(0)
    stop += 1
    return start, stop


def ssim(x, y, crop=True, **kwargs):

    assert x.shape == y.shape

    if crop:
        start_x, stop_x = rectangular_hull(np.isfinite(x))
        start_y, stop_y = rectangular_hull(np.isfinite(y))
        start = np.stack([start_x, start_y]).max(0)
        stop = np.stack([stop_x, stop_y]).min(0)
        x = x[start[0]:stop[0], start[1]:stop[1]]
        y = y[start[0]:stop[0], start[1]:stop[1]]

    x = standardize(x)
    y = standardize(y)
    mask = np.isfinite(x) * np.isfinite(y)
    x[~mask] = 0.0
    y[~mask] = 0.0
    s = structural_similarity(x, y, data_range=1.0-0.0, **kwargs)
    return s


def rmse(x, y):
    x = standardize(x)
    y = standardize(y)
    return np.square(x - y).mean()**0.5


def metric_fin(x, y, method='pearson'):
    mask = np.isfinite(x) * np.isfinite(y)
    x = x[mask]
    y = y[mask]
    method_dict = {
            'pearson': corr_pearson,
            'pearson_stablized': corr_pearson_stablized,
            'spearman': corr_spearman,
            'uncentered': corr_uncentered,
            'psnr': psnr,
            'rmse': rmse,
            }
    method_func = method_dict[method]
    corr = method_func(x, y)
    return corr


def get_indices(x, x_all):
    x_all = np.array(x_all)
    indices = [np.where(x_all == u)[0] for u in x]
    assert all([ind.size == 1 for ind in indices])
    indices = [ind[0] for ind in indices]
    return indices


def plot_collated(x_true, x_pred, outfile):
    mask = np.isfinite(x_true) * np.isfinite(x_pred)
    x_true[~mask] = np.nan
    x_pred[~mask] = np.nan
    x_pred = np.clip(x_pred, np.nanmin(x_true), np.nanmax(x_true))
    x = np.concatenate([x_true, x_pred], axis=1)
    plot_matrix(x, outfile)


def denoise(x, filter_strength=10):
    x = x.copy()
    isin = np.isfinite(x).all(-1)
    xmin, xmax = np.nanmin(x, (0, 1)), np.nanmax(x, (0, 1))
    x -= xmin
    x /= xmax + 1e-12
    x = (x * 255).astype(np.uint8)
    for i in range(x.shape[-1]):
        x[..., i] = fastNlMeansDenoising(x[..., i], h=filter_strength)
    x = x.astype(np.float32)
    x /= 255.0
    x *= xmax
    x += xmin
    x[~isin] = np.nan
    return x


def main():

    prefix = sys.argv[1]  # e.g. data/xenium/rep1/
    factor = int(sys.argv[2])  # e.g. 2

    use_cell_masks = factor >= 9000
    if use_cell_masks:
        i_strata = factor - 9000

    truth = load_pickle(
            prefix+'cnts-truth-agg/radius0004-stride01-square/data.pickle')
    cnts, gene_names = truth['cnts'], truth['gene_names']
    cnts = cnts.astype(np.float32)
    # convert 8x8 px to 16x16 px
    cnts = reduce(
            cnts, '(h1 h) (w1 w) c -> h1 w1 c', 'sum',
            h=2, w=2)

    if use_cell_masks:
        mask_tissue = load_mask(prefix+'mask.png')
        cell_data = load_pickle(
                f'{prefix}cell-level/area{i_strata}/'
                'cell-data.pickle')
        wts_cell = cell_data['weights']
        masks_cell = cell_data['masks']
        cnts_cell = cell_data['counts']

    gene_names_plot = ['ERBB2', 'ESR1', 'MS4A1', 'ITGAX', 'PGR']
    eval_list = []
    for i, gname in enumerate(gene_names):

        ct_pred = load_pickle(
                f'{prefix}cnts-super/{gname}.pickle', verbose=False)
        ct_pred = standardize(ct_pred)

        if not use_cell_masks:
            ct = cnts[..., i]
            ct = reduce(
                    ct, '(h1 h) (w1 w) -> h1 w1', 'sum',
                    h=factor, w=factor)
            ct_pred = reduce(
                    ct_pred, '(h1 h) (w1 w) -> h1 w1', 'sum',
                    h=factor, w=factor)
        else:
            ct = cnts_cell[gname].to_numpy()
            ct_pred = aggregate_by_weights(ct_pred, wts_cell)

        ct, ct_pred = standardize(ct), standardize(ct_pred)

        if gname in gene_names_plot:

            plotfile = (
                    f'{prefix}cnts-super-eval/factor{factor:04d}/'
                    f'plots/truth/{gname}.png')
            plotfile_pred = (
                    f'{prefix}cnts-super-eval/factor{factor:04d}/'
                    f'plots/pred/{gname}.png')

            if not use_cell_masks:
                plot_matrix(
                        ct,
                        plotfile,
                        white_background=True)
                plot_matrix(
                        ct_pred,
                        plotfile_pred,
                        white_background=True)
            else:
                plot_cells(ct, masks_cell, mask_tissue, plotfile)
                plot_cells(
                        ct_pred, masks_cell, mask_tissue,
                        plotfile_pred)

        eval = {}
        eval['pearson'] = metric_fin(ct, ct_pred, 'pearson')
        eval['rmse'] = metric_fin(ct, ct_pred, 'rmse')
        if not use_cell_masks:
            eval['ssim'] = ssim(ct, ct_pred)
        eval_list.append(eval)
    df = {ke: [e[ke] for e in eval_list] for ke in eval_list[0].keys()}
    df = pd.DataFrame(df)
    df.index = gene_names
    df.index.name = 'gene'
    df = df.round(4)
    save_tsv(df, f'{prefix}cnts-super-eval/factor{factor:04d}.tsv')


if __name__ == '__main__':
    main()
