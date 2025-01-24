from itertools import product
import multiprocessing
import sys
import os

import numpy as np
from einops import rearrange

from utils import (
        load_tsv, load_pickle, load_image, read_lines,
        load_mask, save_pickle)
from visual import mat_to_img, save_image
from stitch_eval import (
        annotate, draw_border, reduce_matrix, repeat_matrix,
        metrics_to_str, add_colorbar)
from evaluate_imputed import ssim, metric_fin
from image import crop_image


def shorten_factor_name(s):
    i, t = s.split('x ')
    i = int(i)
    return f'{t}{i:03d}x'


def process(im, title, title_sub, title_cbar, shape):
    extent = [(0, s) for s in shape]
    im = crop_image(im, extent, mode='constant', constant_values=255)
    im = add_colorbar(im, 'turbo', title_cbar)
    im = annotate(im, title_sub)
    im = annotate(im, title)
    im = draw_border(im)
    return im


def select_genes(filename, n_genes, n_groups):
    cnts_train = load_tsv(filename)
    gene_names = cnts_train.var(0).sort_values().index
    gene_names = gene_names.to_numpy()[::-1]
    indices = np.linspace(0, 1, n_genes) * (len(gene_names) - 1)
    indices = np.round(indices).astype(int)
    gene_names = gene_names[indices]
    gene_names = gene_names.reshape(n_groups, -1)
    return gene_names


def load_truth(prefix, factor=1, factor_backward=1, convert=False):
    data = load_pickle(
            f'{prefix}cnts-truth-agg/'
            'radius0004-stride01-square/data.pickle',
            verbose=False)
    cnts, gene_names = data['cnts'], data['gene_names']
    # convert 8x8 px truth to 16x16 px
    cnts = reduce_matrix(cnts, 2)
    if factor is not None and factor > 1:
        cnts = reduce_matrix(cnts, factor)
        cnts = repeat_matrix(cnts, int(factor*factor_backward))
    cnts = cnts.transpose(2, 0, 1)
    if convert:
        cnts = [mat_to_img(ct) for ct in cnts]
    out = {
            gname: ct
            for gname, ct in zip(gene_names, cnts)}
    return out


def load_training(filename, factor=1):
    img = load_image(filename, verbose=False)
    img = img.astype(float)
    # spot plots resolution: 4x4 px
    factor *= 4
    img = reduce_matrix(img, factor)
    img = img.astype(np.uint8)
    return img


def plot_concise():

    # prefix = 'data/'
    # dataname = 'xenium/'
    # train_file = 'data/xenium/rep12/cnts.tsv'
    # tasks = ['', 'oos/']
    # methods = ['', 'xfuse/']
    # sections = ['rep1/', 'rep2/']
    # n_genes = 40
    # n_groups = 5
    # outpref = 'results/concise/xenium/'

    prefix = 'data/'
    dataname = 'xenium-mouse-brain/'
    train_file = 'data/xenium-mouse-brain/cnts.tsv'
    tasks = ['']
    methods = ['', 'xfuse/']
    meth_name = {
            '': 'istar',
            'xfuse/': 'xfuse'}
    sections = ['']
    n_genes = 120
    n_groups = 15
    outpref = 'results/concise/xenium-mouse-brain/'

    factor = 4  # prediction resolution
    factor_backward = 1  # visualization resolution
    gene_groups = select_genes(train_file, n_genes, n_groups)

    truths = {
            sec: load_truth(
                f'{prefix}{dataname}{sec}',
                factor=factor, factor_backward=factor_backward,
                convert=True)
            for sec in sections}
    truths_mat = {
            sec: load_truth(
                f'{prefix}{dataname}{sec}',
                factor=factor, factor_backward=factor_backward,
                convert=False)
            for sec in sections}

    # group x sec x gene x (truth + train + method x task)

    mat_dict = {}

    for (grp, genes), sec in product(enumerate(gene_groups), sections):

        mask_filename = f'{prefix}{dataname}{sec}mask-small.png'
        if os.path.exists(mask_filename):
            mask = load_mask(
                    mask_filename,
                    verbose=False)
        else:
            mask = None

        image_list = []
        for ge in genes:

            img_list = []
            mat_dict[ge] = {}

            train = load_image(
                    f'{prefix}{dataname}{sec}spots/{ge}.png',
                    verbose=False)
            train = reduce_matrix(train, 4)
            img_list.append(train)

            tru = truths[sec][ge]
            img_list.append(tru)
            mat_dict[ge]['truth'] = truths_mat[sec][ge]

            for meth, tas in product(methods, tasks):
                inpfile = (
                        f'{prefix}{meth}{dataname}{tas}{sec}'
                        f'cnts-super/{ge}.pickle')
                x = load_pickle(inpfile, verbose=False)
                x = reduce_matrix(x, factor)
                x = repeat_matrix(x, factor*factor_backward)
                if mask is not None:
                    x[~mask] = np.nan
                mat_dict[ge][meth_name[meth]] = x
                x = mat_to_img(x)
                img_list.append(x)

            img_list = [draw_border(im) for im in img_list]
            img = np.concatenate(img_list)

            img = annotate(
                    img, ge, text_height=256/img.shape[0],
                    text_start=(0.2, 0.7), thickness=10)
            image_list.append(img)
        image = np.concatenate(image_list, 1)
        save_image(image, f'{outpref}{sec}group{grp:02d}.png')
        save_pickle(
                mat_dict,
                f'{outpref}results.pickle')


def make_gallery_single_factor(
        gene, factor, methods, sections, tasks, prefix, dataname,
        truths, shape):
    imgs = []
    for sec_name, sec_path in sections.items():

        mask_filename = f'{prefix}{dataname}{sec_path}mask-small.png'
        if os.path.exists(mask_filename):
            mask = load_mask(
                    mask_filename,
                    verbose=False)
        else:
            mask = None

        train = load_image(
                f'{prefix}{dataname}{sec_path}spots/{gene}.png',
                verbose=False)
        train = reduce_matrix(train, 4)
        enh = 128 // factor**2
        enh_note = f'{enh}x enhancement'
        imgs.append({
            'title': '',
            'title_sub': (
                'Spot-level gene expression, '
                f'{sec_name} ({enh_note})'),
            'img': train})

        tru = truths[sec_name][gene]
        tru = reduce_matrix(tru, factor)
        tru_img = mat_to_img(repeat_matrix(tru, factor))
        imgs.append({
            'title': '',
            'title_sub': f'Ground truth, {sec_name} ({enh_note})',
            'img': tru_img})

        for (
                (meth_name, meth_path),
                (tas_name, tas_path)
                ) in product(
                        methods.items(),
                        tasks.items()):
            inpfile = (
                    f'{prefix}{meth_path}{dataname}{tas_path}'
                    f'{sec_path}cnts-super/{gene}.pickle')
            x = load_pickle(inpfile, verbose=False)
            x = reduce_matrix(x, factor)

            metrics = {
                    'RMSE': metric_fin(tru, x, 'rmse'),
                    'SSIM': ssim(tru, x),
                    'PCC': metric_fin(tru, x, 'pearson'),
                    }
            metrics = metrics_to_str(metrics)

            x = repeat_matrix(x, factor)
            if mask is not None:
                x[~mask] = np.nan
            x = mat_to_img(x)

            title = (
                    f'{meth_name}, {sec_name}, {tas_name} '
                    f'({enh_note})')
            imgs.append({
                'title': title,
                'title_sub': metrics,
                'img': x})

    imgs = [
            process(
                im=im['img'], title=im['title'],
                title_sub=im['title_sub'], title_cbar=gene,
                shape=shape)
            for im in imgs]
    imgs = np.array(imgs)

    n_tasks = len(tasks)
    n_methods = len(methods)
    n_sections = len(sections)
    if n_tasks == 1 and n_methods == 2:
        image = rearrange(
                imgs, '(t m) h w c -> (t h) (m w) c',
                m=2, t=2)
    else:
        image = rearrange(
                imgs, '(s m t) h w c -> (m h) (s t w) c',
                s=n_sections, m=n_methods+1, t=n_tasks)

    image = image[65:]  # remove whitespace in first row
    save_image(
            image,
            f'results/{dataname}combined/'
            f'enhancement{enh:03d}x/{gene}.png')


def make_gallery(factors, **kwargs):
    for fac in factors:
        make_gallery_single_factor(factor=fac, **kwargs)


def make_gallery_kwargs(kwargs):
    make_gallery(**kwargs)


def plot_all():

    dataname = sys.argv[1]

    n_jobs = 1

    if dataname == 'xenium/':
        prefix = 'data/'
        genes_filename = 'data/xenium/rep12/gene-names.txt'
        tasks = {'In-sample': '', 'Out-of-sample': 'oos/'}
        methods = {'iStar': '', 'XFuse': 'xfuse/'}
        sections = {'Section 1': 'rep1/', 'Section 2': 'rep2/'}
        factors = [1, 2, 4, 8]
        shape = (920, 1280)
    elif dataname == 'xenium-mouse-brain/':
        prefix = 'data/'
        tasks = {'In-sample': ''}
        methods = {'iStar': '', 'XFuse': 'xfuse/'}
        sections = {'Section 1': ''}
        factors = [1, 2, 4, 8]
        shape = (768, 1152)
        genes_filename = 'data/xenium-mouse-brain/gene-names.txt'
    else:
        raise ValueError('Dataset name not recognized')

    genes = read_lines(genes_filename)

    truths = {
            sec_name: load_truth(
                f'{prefix}{dataname}{sec_path}')
            for sec_name, sec_path in sections.items()}

    kwargs_list = [
            dict(
                gene=ge, factors=factors, methods=methods,
                sections=sections, tasks=tasks, prefix=prefix,
                dataname=dataname, truths=truths, shape=shape)
            for ge in genes]
    if n_jobs == 1:
        for kwargs in kwargs_list:
            make_gallery_kwargs(kwargs)
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            pool.map(make_gallery_kwargs, kwargs_list)


def main():
    plot_concise()
    plot_all()


if __name__ == '__main__':
    main()
