import numpy as np
from einops import rearrange

from utils import load_pickle, load_mask
from cluster import preprocess_and_cluster
from image import smoothen
from integrate_features import integrate_features


def get_data_separately():

    prefix_list = [
            'data/3d/her2st/G1/',
            'data/3d/her2st/G2/',
            'data/3d/her2st/G3/']

    mode = 'hist'

    if mode == 'hist':
        embs_list = [
                load_pickle(pref+'embeddings-hist.pickle')
                for pref in prefix_list]
        x_cls = np.stack([embs['cls'] for embs in embs_list], 1)
        x_sub = np.stack([embs['sub'] for embs in embs_list], 1)
        x = integrate_features((x_cls, x_sub))
        x = x.transpose(1, 2, 3, 0)
    elif mode == 'gene':
        embs_list = [
                load_pickle(pref+'embeddings-gene.pickle')
                for pref in prefix_list]
        x = np.stack([embs['cls'] for embs in embs_list])
        x = x.transpose(0, 2, 3, 1)
    else:
        raise ValueError('mode not recognized')

    mask_list = [
            load_mask(pref+'mask-small.png') for pref in prefix_list]
    mask = np.stack(mask_list)

    return x, mask


def get_data(
        prefix, n_sections, filter_size=None,
        include_histology=False):

    embs_gene = load_pickle(prefix+'embeddings-gene.pickle')
    x = np.stack(embs_gene['cls']).astype(np.float32)

    if filter_size is not None:
        x = np.stack([smoothen(u, filter_size) for u in x])

    if include_histology:
        embs_hist = load_pickle(prefix+'embeddings-hist.pickle')
        x_hist = np.stack(embs_hist['cls']).astype(np.float32)
        x = integrate_features(
                xs=(x, x_hist),
                n_components_inp=(0.9, 0.9),
                n_components_out=0.95)

    x = rearrange(x, 'c (d h) w -> d h w c', d=n_sections)

    mask = load_mask(prefix+'mask-small.png')
    mask = rearrange(mask, '(d h) w -> d h w', d=n_sections)

    return x, mask


def interpolate_two(x_start, x_stop, n):
    dtype = x_start.dtype
    w = np.linspace(0, 1, n+2, dtype=dtype)
    w = np.expand_dims(w, (np.arange(x_start.ndim)+1).tolist())
    y = x_start * (1-w) + x_stop * w
    return y


def interpolate(xs, n_between):

    storey_list = []
    for i in range(len(xs)-1):
        storey = interpolate_two(xs[i], xs[i+1], n_between)
        if i > 0:
            storey = storey[1:]
        storey_list.append(storey)
    y = np.concatenate(storey_list)

    return y


def interpolate_labs(labs, n_between):
    probs, labs_min = labs_to_probs(labs)
    probs = interpolate(probs, n_between)
    labs_3d = probs_to_labs(probs, labs_min)
    return labs_3d


def cluster_3d(x, **kwargs):

    depth = x.shape[0]
    x = rearrange(x, 'd h w c -> c (d h) w')

    labels_list = preprocess_and_cluster(x, **kwargs)
    labels = labels_list[0]

    labels = rearrange(labels, '(d h) w -> d h w', d=depth)

    return labels


def labs_to_probs(labs):
    lmin, lmax = labs.min(), labs.max()+1
    probs = [labs == la for la in range(lmin, lmax+1)]
    probs = np.stack(probs, -1)
    probs = probs.astype(np.float32)
    return probs, lmin


def probs_to_labs(probs, labs_min=0):
    labs = probs.argmax(-1)
    labs += labs_min
    return labs


def cluster(x, **kwargs):
    x = x.transpose(2, 0, 1)
    labs = preprocess_and_cluster(x, **kwargs)
    return labs


def main():

    prefix = 'data/3d/her2st/G123/'
    n_sections = 3
    n_between = 3

    filter_size = 8
    include_histology = False

    x, mask = get_data(
            prefix,
            n_sections=n_sections,
            filter_size=filter_size,
            include_histology=include_histology)

    x = interpolate(x, n_between)
    mask = interpolate(
            mask[..., np.newaxis].astype(np.float32),
            n_between)[..., 0] > 0.5

    x[~mask] = np.nan

    labels = cluster_3d(
            x, method='km', n_clusters=[10, 20, 30],
            min_cluster_size=20,
            prefix=prefix+'clusters-gene/')
    print(labels.shape)


if __name__ == '__main__':
    main()
