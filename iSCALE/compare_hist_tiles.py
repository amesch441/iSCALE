import argparse
from itertools import product
import os
from time import time

import numpy as np
from einops import reduce
import matplotlib
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

from utils import load_pickle, save_pickle, load_mask, load_tsv
from reduce_dim import reduce_dim
from cluster import cluster
from visual import plot_labels, plot_matrix, cmap_tab70
from stitch_sections import stitch_images, get_inputs
from stitch_sections import save_extent as get_extents


np.random.seed(0)


def get_coords(labels):
    coords = np.full(labels.shape + (2,), np.nan)
    for lab in range(labels.max() + 1):
        isin = labels == lab
        indices = np.stack(np.where(isin), -1)
        imin, imax = indices.min(0), indices.max(0)
        c = np.stack(np.meshgrid(
                np.linspace(0, 1, imax[0] - imin[0]),
                np.linspace(0, 1, imax[1] - imin[1]),
                indexing='ij'), -1)
        coords[imin[0]:imax[0], imin[1]:imax[1]] = c
    return coords


def get_mask_extents(data):
    name_list = list(data.keys())
    mask_list = [data[name]['mask'] for name in name_list]
    mask, origins, shapes = stitch_images(mask_list, axis=0)
    extents = get_extents(origins, shapes)
    return mask, extents


def aggregate_distances(dist_mat):
    return (dist_mat**2).mean()**0.5


def plot_embs(x, mask, prefix):
    y = np.full(mask.shape, np.nan, dtype=x.dtype)
    for i in range(x.shape[-1]):
        y[mask] = x[..., i]
        plot_matrix(y, f'{prefix}-{i:03d}.png')


def plot_clusters(x, mask, filename):
    y = np.full(mask.shape, -1, dtype=x.dtype)
    y[mask] = x
    plot_labels(y, filename)


def plot_dists(
        dists_list, labels, mask, filename, hide_diagonal=False):
    x_list = []
    for i, dists in enumerate(dists_list):
        x = np.full(labels.shape, np.nan)
        for j, dist in enumerate(dists):
            mask_lab = labels == j
            if hide_diagonal and j == i:
                dist = np.full_like(dist, np.nan)
            x[mask_lab*mask] = dist
        x_list.append(x)
    xs = np.concatenate(x_list, axis=1)
    plot_matrix(xs, filename)


def get_starts_stops(extents):
    starts = extents[['origin_y', 'origin_x']].to_numpy()
    lengths = extents[['length_y', 'length_x']].to_numpy()
    stops = starts + lengths
    return starts, stops


def get_section_labels(shape, extents):
    labels = np.full(shape, -1)
    starts, stops = get_starts_stops(extents)
    for i, (sta, sto) in enumerate(zip(starts, stops)):
        labels[sta[0]:sto[0], sta[1]:sto[1]] = i
    return labels


def get_neighborhoods(x, n_neighbors):
    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(x)
    return model


def compute_distance_pairwise(
        x_all, labels, n_spots_list=None, n_neighbors=5):

    n_sections = labels.max() + 1

    dist_mean = np.full((n_sections, n_sections), np.nan)
    dists = [
            [None for __ in range(n_sections)]
            for __ in range(n_sections)]
    for i0, i1 in product(range(n_sections), range(n_sections)):
        x0 = x_all[labels == i0]
        x1 = x_all[labels == i1]
        if n_spots_list is None:
            n_spots = None
        else:
            n_spots = n_spots_list[i0]
        dist = compute_distance(
                x0, x1, n_spots=n_spots, n_neighbors=n_neighbors)
        dists[i0][i1] = dist
        dist_mean[i0, i1] = aggregate_distances(dist)

    return dists, dist_mean


def compute_distance(x0, x1, n_spots=None, n_neighbors=5):

    if n_spots is not None:
        n_samples = min(n_spots, len(x0))
        idx = np.random.choice(len(x0), n_samples, replace=False)
        x0 = x0[idx]
    model = get_neighborhoods(x0, n_neighbors=n_neighbors)

    dist, __ = model.kneighbors(x1)
    dist = (dist**2).mean(1)**0.5  # mean distance of all neighbors

    return dist


def plot_scatter(x, labels, prefix, color=None):
    plot_scatter_overlay(
            x, labels=labels,
            filename=prefix+'embeddings-overlay.png')
    plot_scatter_individual(
            x, labels=labels, color=color,
            filename=prefix+'embeddings-individual.png')


def plot_scatter_individual_coords(
        x_lowdim, labels, coords, prefix):
    cmap = matplotlib.colormaps.get_cmap('PiYG_r')
    for i in range(coords.shape[-1]):
        color = cmap(coords[..., i])
        filename = f'{prefix}{i}.png'
        plot_scatter_individual(
                x_lowdim, labels=labels,
                color=color,
                filename=filename)


def plot_scatter_individual(x, labels, filename, color=None):
    n_cols = 5
    uniq_labs = np.unique(labels)
    n_rows = (len(uniq_labs) + n_cols - 1) // n_cols
    minmax = x.min(0), x.max(0)
    offset = (minmax[1] - minmax[0]) * 0.05
    plt.figure(figsize=(n_cols*10, n_rows*10))

    for i in uniq_labs:
        plt.subplot(n_rows, n_cols, 1+i)
        isin = i == labels
        if color is None:
            c = None
        else:
            c = color[isin]
        plt.scatter(x[isin, 0], x[isin, 1], c=c)
        plt.title(f'Section {i:02d}', alpha=0.2)
        plt.xlim(minmax[0][0]-offset[0], minmax[1][0]+offset[0])
        plt.ylim(minmax[0][1]-offset[1], minmax[1][1]+offset[1])
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(filename)


def plot_scatter_overlay(x, labels, filename):
    cmap = cmap_tab70
    plt.figure(figsize=(20, 20))
    for lab in np.unique(labels):
        isin = labels == lab
        color = cmap(lab)
        plt.scatter(
                x[isin, 0], x[isin, 1],
                color=color, label=f'Section {lab}', alpha=0.2)
    plt.legend(
            loc='upper left',
            bbox_to_anchor=(1.05, 1.00),
            ncol=3)
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(filename)


def get_embeddings(prefix, factor_tiles):

    t0 = time()
    embs = load_pickle(prefix+'embeddings-hist.pickle')
    print(int(time()-t0), 'sec')
    mask = load_mask(prefix+'mask-small.png')
    print(int(time()-t0), 'sec')
    x = np.stack(embs['cls'], -1)

    x = reduce(
            x, '(h0 h1) (w0 w1) c -> h0 w0 c', 'mean',
            h1=factor_tiles, w1=factor_tiles)
    mask = reduce(
            mask, '(h0 h1) (w0 w1) -> h0 w0', 'min',
            h1=factor_tiles, w1=factor_tiles)

    x[~mask] = np.nan

    return x, mask


def load_data_independent(
        filename, factor, sample_spots=False, cache=None):
    names, prefixs = get_inputs(filename)
    nameprefs = list(zip(names, prefixs))
    if cache is not None and os.path.exists(cache):
        print('Loading cache...')
        data = load_pickle(cache)
    else:
        data = {name: {} for name in names}
        for (i, (name, pref)) in enumerate(nameprefs):
            x, mask = get_embeddings(pref, factor)
            if sample_spots:
                locs = load_tsv(pref+'locs.tsv')
                n_spots = locs.shape[0]
            else:
                n_spots = None
            data[name]['x'] = x
            data[name]['mask'] = mask
            data[name]['n_spots'] = n_spots
    return data


def load_data_concatenated(prefix, factor, sample_spots=False):

    x, mask = get_embeddings(prefix, factor)

    extents = load_tsv(prefix+'extents.tsv')
    extents //= factor * 16  # extents are in hist pixel scale
    name_list = extents.index.to_list()

    if sample_spots:
        locs = load_tsv(prefix+'locs.tsv')
        spot_sec_names = np.array(
                [e[0] for e in locs.index.str.split('_')])
        n_spots_list = [
                (spot_sec_names == name).sum()
                for name in name_list]
    else:
        n_spots_list = [None] * len(name_list)

    starts, stops = get_starts_stops(extents)

    data = {name: {} for name in name_list}
    for i, name in enumerate(name_list):
        sta, sto = starts[i], stops[i]
        data[name]['x'] = x[sta[0]:sto[0], sta[1]:sto[1]]
        data[name]['mask'] = mask[sta[0]:sto[0], sta[1]:sto[1]]
        data[name]['n_spots'] = n_spots_list[i]

    return data


def compute_distance_multi(data, n_neighbors):
    name_list = data.keys()
    n_sections = len(name_list)
    dist_mean = np.full((n_sections, n_sections), np.nan)
    dists = [
            [None for __ in range(n_sections)]
            for __ in range(n_sections)]
    t0 = time()
    for i0, name0 in enumerate(name_list):
        for i1, name1 in enumerate(name_list):
            x0 = data[name0]['x']
            x1 = data[name1]['x']
            mask0 = data[name0]['mask']
            mask1 = data[name1]['mask']
            n_spots = data[name0]['n_spots']

            dist = compute_distance(
                    x0[mask0], x1[mask1],
                    n_spots=n_spots, n_neighbors=n_neighbors)
            dists[i0][i1] = dist
            dist_mean[i0, i1] = aggregate_distances(dist)
        print(i0, int(time() - t0), 'sec')
    return dists, dist_mean


def analyze_joint(data, prefix):

    mask, extents = get_mask_extents(data)
    labels = get_section_labels(mask.shape, extents)

    x = np.concatenate([
        data[name]['x'][data[name]['mask']]
        for name in data.keys()])

    cache_filename = prefix+'joint.pickle'

    if os.path.exists(cache_filename):
        print('Loading cache...')
        cache = load_pickle(cache_filename)
        x_lowdim, clusters = cache['embeddings'], cache['clusters']
    else:
        x_lowdim, __ = reduce_dim(x, method='umap', n_components=2)
        clusters, __ = cluster(x.T, n_clusters=10, method='km')
        save_pickle(
                dict(embeddings=x_lowdim, clusters=clusters),
                cache)

    labels_flat = labels[mask]

    # plot_scatter(
    #         x_lowdim, labels=labels_flat, color=clusters,
    #         prefix=prefix)
    plot_scatter_overlay(
            x_lowdim, labels=labels_flat,
            filename=prefix+'embeddings-overlay.png')
    plot_scatter_individual(
            x_lowdim, labels=labels_flat,
            color=cmap_tab70(clusters),
            filename=prefix+'embeddings-individual-clusters.png')

    coords = get_coords(labels)
    coords_flat = coords[mask]
    plot_scatter_individual_coords(
            x_lowdim, labels=labels_flat,
            coords=coords_flat,
            prefix=prefix+'embeddings-individual-coords')

    plot_clusters(
            clusters, mask,
            filename=prefix+'clusters.png')
    plot_embs(
            x_lowdim, mask,
            prefix=prefix+'embeddings')


def analyze_pairwise(
        data, n_neighbors, prefix, hide_diagonal=True):
    dists, dist_mean = compute_distance_multi(data, n_neighbors)
    save_pickle(
            dict(dists=dists, dist_mean=dist_mean),
            prefix+'pairwise.pickle')
    mask, extents = get_mask_extents(data)
    labels = get_section_labels(mask.shape, extents)
    plot_dists(
            dists, labels, mask,
            hide_diagonal=hide_diagonal,
            filename=prefix+'distances.png')
    if hide_diagonal:
        np.fill_diagonal(dist_mean, np.nan)
    plot_matrix(dist_mean, prefix+'distances-mean.png')


def compare_hist_tiles(
        prefix, sections_are_concatenated=False,
        factor=16, sample_spots=False,
        n_neighbors=5):
    if sections_are_concatenated:
        data = load_data_concatenated(
                prefix, factor=factor,
                sample_spots=sample_spots)
    else:
        cache = f'{prefix}cache-factor{factor:02d}.pickle'
        data = load_data_independent(
                prefix+'sections.txt', factor=factor,
                sample_spots=sample_spots, cache=cache)
    analyze_joint(data, prefix+'sectosec/')
    analyze_pairwise(
        data, n_neighbors, prefix+'sectosec/', hide_diagonal=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--concatenated', action='store_true')
    parser.add_argument('--factor', type=int, default=16)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    compare_hist_tiles(
            args.prefix,
            sections_are_concatenated=args.concatenated,
            factor=args.factor)


if __name__ == '__main__':
    main()
