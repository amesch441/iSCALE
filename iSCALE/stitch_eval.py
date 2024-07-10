import sys

import numpy as np
from einops import reduce, repeat
import matplotlib.pyplot as plt
import cv2 as cv

from utils import load_pickle, load_image
from stitch_images import stitch_images
from utils import save_image
from visual import mat_to_img
from evaluate_imputed import ssim, metric_fin


def metrics_to_str(dic):
    x = [f'{ke}: {va:.2f}' for ke, va in dic.items()]
    return ', '.join(x)


def reduce_matrix(x, factor):
    small = 'h0 w0'
    large = '(h0 h1) (w0 w1)'
    if x.ndim == 3:
        small = f'{small} c'
        large = f'{large} c'
    dtype = x.dtype
    x = x.astype(np.float32)
    x = reduce(
            x, f'{large} -> {small}', 'mean',
            h1=factor, w1=factor)
    x = x.astype(dtype)
    return x


def repeat_matrix(x, factor):
    small = 'h0 w0'
    large = '(h0 h1) (w0 w1)'
    if x.ndim == 3:
        small = f'{small} c'
        large = f'{large} c'
    x = repeat(
            x, f'{small} -> {large}',
            h1=factor, w1=factor)
    return x


def draw_border(img, thickness=4):
    img = img.copy()
    img[:thickness] = 0
    img[-thickness:] = 0
    img[:, :thickness] = 0
    img[:, -thickness:] = 0
    return img


def add_colorbar(img, cmap, label):
    height = img.shape[0] // 16
    cmap = plt.get_cmap(cmap)
    cbar = cmap(np.linspace(0, 1, img.shape[1]))[..., :3]
    cbar = (cbar * 255).astype(np.uint8)
    cbar = cbar + np.zeros_like(img[:height])
    cbar = cv.putText(
            img=cbar, text='min',
            org=(int(cbar.shape[1]*0.02), int(cbar.shape[0]*0.7)),
            color=(255, 255, 255),
            fontScale=cbar.shape[0]*0.020,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, thickness=2)
    cbar = cv.putText(
            img=cbar, text='max',
            org=(int(cbar.shape[1]*0.92), int(cbar.shape[0]*0.7)),
            color=(255, 255, 255),
            fontScale=cbar.shape[0]*0.020,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, thickness=2)
    cbar = cv.putText(
            img=cbar, text=label,
            org=(int(cbar.shape[1]*0.45), int(cbar.shape[0]*0.7)),
            color=(0, 0, 0),
            fontScale=cbar.shape[0]*0.020,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, thickness=2)
    img = np.concatenate([cbar, img])
    return img


def combine_images(imgs, labels, gene_name):

    cmap = 'turbo'

    if isinstance(imgs, list):
        imgs = np.stack(imgs)
    # mask = np.isfinite(imgs)
    # imgs -= np.nanmin(imgs, (1, 2), keepdims=True)
    # imgs /= np.nanmax(imgs, (1, 2), keepdims=True) + 1e-12
    # imgs = plt.get_cmap(cmap)(imgs)[..., :3]
    # imgs[~mask] = 1.0
    # imgs = (imgs * 255).astype(np.uint8)
    imgs = [add_colorbar(im, cmap, gene_name) for im in imgs]
    imgs = [annotate(im, lab) for im, lab in zip(imgs, labels)]
    imgs = [draw_border(im) for im in imgs]
    im = stitch_images(imgs, axis=None)
    return im


def annotate(
        img, text,
        text_height=0.0625, text_start=(0.02, 0.8), thickness=2):
    text_height = int(text_height * img.shape[0])
    pad = np.zeros_like(img[:text_height]) + np.nanmax(img)
    text_width = pad.shape[1]
    org = (int(text_width*text_start[0]), int(text_height*text_start[1]))
    pad = cv.putText(
            img=pad, text=text, org=org, color=(0, 0, 0),
            fontScale=text_height*0.020,
            fontFace=cv.FONT_HERSHEY_SIMPLEX, thickness=thickness)
    img = np.concatenate([pad, img])
    return img


def main():

    dataname = sys.argv[1]  # e.g. xenium/rep1/
    factor = int(sys.argv[2])  # e.g. 2

    truth = load_pickle(
            f'data/{dataname}cnts-truth-agg/'
            'radius0004-stride01-square/data.pickle')
    cnts, gene_names = truth['cnts'], truth['gene_names']
    cnts = cnts.astype(np.float32)

    for ct_true, gname in zip(cnts.transpose(2, 0, 1), gene_names):

        # load predictions
        ct_ours = load_pickle(
                f'data/{dataname}cnts-super/{gname}.pickle', verbose=False)
        ct_base = load_pickle(
                f'data/xfuse/{dataname}cnts-super/{gname}.pickle',
                verbose=False)

        train = load_image(f'data/{dataname}spots/{gname}.png')

        # rescale image
        ct_true = reduce_matrix(ct_true, factor*2)
        ct_ours = reduce_matrix(ct_ours, factor)
        ct_base = reduce_matrix(ct_base, factor)
        train = reduce_matrix(train, factor*4)

        # compute metrics
        metrics_ours = {
                'RMSE': metric_fin(ct_true, ct_ours, 'rmse'),
                'SSIM': ssim(ct_true, ct_ours),
                'PCC': metric_fin(ct_true, ct_ours, 'pearson'),
                }
        metrics_base = {
                'RMSE': metric_fin(ct_true, ct_base, 'rmse'),
                'SSIM': ssim(ct_true, ct_base),
                'PCC': metric_fin(ct_true, ct_base, 'pearson'),
                }

        # restore original image scale
        ct_true = repeat_matrix(ct_true, factor)
        ct_ours = repeat_matrix(ct_ours, factor)
        ct_base = repeat_matrix(ct_base, factor)
        train = repeat_matrix(train, factor)

        ct_true = mat_to_img(ct_true)
        ct_ours = mat_to_img(ct_ours)
        ct_base = mat_to_img(ct_base)

        # combine results into a single image
        s = factor * 16
        notes_truth = f'1 superpixel = {s}x{s} pixels = {s//2}um x {s//2}um'
        img = combine_images(
                [ct_true, train, ct_ours, ct_base],
                [
                    'Ground truth (' + notes_truth + ')',
                    'Spot-level gene expression',
                    'iStar (' + metrics_to_str(metrics_ours) + ')',
                    'XFuse (' + metrics_to_str(metrics_base) + ')'],
                gname
                )
        save_image(
                img,
                f'results/{dataname}cnts-super-eval/factor{factor:04d}/'
                f'plots/combined/{gname}.png'
                )


if __name__ == '__main__':
    main()
