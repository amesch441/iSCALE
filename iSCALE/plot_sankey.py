import sys

import numpy as np
import plotly.graph_objects as go

from utils import load_pickle
from visual import cmap_tab70, plot_labels


def get_labels(prefix, n_clusters_list):
    labels = [
            load_pickle(f'{prefix}nclusters{n:03d}/labels.pickle')
            for n in n_clusters_list]
    labels = np.stack(labels, -1)
    return labels


def get_nodes_edges(x):
    uniqs, counts = np.unique(x, axis=0, return_counts=True)
    source = uniqs[:, 0]
    target = uniqs[:, 1]
    value = counts
    n_clusters = uniqs.max(0) + 1
    name = [[f'{i:02d}/{n:02d}' for i in range(n)] for n in n_clusters]
    cmap = cmap_tab70
    color = [cmap(np.arange(n)) for n in n_clusters]
    return dict(
            source=source, target=target, value=value,
            name=name, color=color)


def get_multi_layers(labels):
    isin = (labels >= 0).all(-1)
    labels = labels[isin]
    out = []
    n_layers = labels.shape[-1]
    for i in range(n_layers-1):
        d = get_nodes_edges(
                labels[:, [i, i+1]])
        out.append(d)
    return out


def unify(nodes_edges):
    n_clusters_list = [ne['source'].max()+1 for ne in nodes_edges]
    n_clusters_list.append(nodes_edges[-1]['target'].max()+1)
    start_list = np.cumsum([0]+n_clusters_list)
    for i, ne in enumerate(nodes_edges):
        ne['source'] += start_list[i]
        ne['target'] += start_list[i+1]


def flatten(nodes_edges):
    out = {}
    for key in ['source', 'target', 'value']:
        out[key] = np.concatenate([ne[key] for ne in nodes_edges])
    for key in ['name', 'color']:
        tocat = [ne[key][0] for ne in nodes_edges]
        tocat += [nodes_edges[-1][key][1]]
        out[key] = np.concatenate(tocat)
    return out


def rgbastr(color):
    return f'rgba({",".join([str(int(c*255)) for c in color])})'


def plot_sankey(source, target, value, name, color, filename):
    # convert rgba color vector to string
    color = [rgbastr(c) for c in color]
    fig = go.Figure(data=[go.Sankey(
        node=dict(label=name, color=color),
        link=dict(source=source, target=target, value=value),
        textfont=go.sankey.Textfont(size=20))])
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    fig.write_image(filename, width=3000, height=2000)
    print(filename)


def get_overlap_matrix(labs_src, labs_tar):
    uniqs_src, cnts_src = np.unique(labs_src, return_counts=True)
    uniqs_tar, cnts_tar = np.unique(labs_tar, return_counts=True)
    overlap = np.full((len(uniqs_src), len(uniqs_tar)), np.nan)
    for i_src, c_src in zip(uniqs_src, cnts_src):
        for i_tar, c_tar in zip(uniqs_tar, cnts_tar):
            c_comm = ((labs_src == i_src) * (labs_tar == i_tar)).sum()
            dice_index = 2 * c_comm / (c_src + c_tar)
            overlap[i_src, i_tar] = dice_index
    return overlap


def align_labels(labs_src, labs_tar):

    # get_similarity matrix
    similarity = get_overlap_matrix(labs_src, labs_tar)

    # match labels based on similarity
    matches = {}
    while len(matches) < min(similarity.shape):
        i_src, i_tar = np.unravel_index(
                np.nanargmax(similarity), similarity.shape)
        matches[i_src] = i_tar
        similarity[i_src, :] = np.nan
        similarity[:, i_tar] = np.nan
    n_src = similarity.shape[0]
    labs_new = np.full(n_src, -1)
    for i_src in range(n_src):
        if i_src in matches.keys():
            labs_new[i_src] = matches[i_src]

    # assign remaining source labels to unmatched source labels
    is_unmatched = labs_new < 0
    if is_unmatched.any():
        vals_unmatched = np.arange(labs_new.max()+1, labs_new.size)
        labs_new[is_unmatched] = vals_unmatched

    transform_dict = {i: lab for i, lab in enumerate(labs_new)}

    return transform_dict


def rearrange_labels_multi(labels):
    isin = (labels >= 0).all(-1)
    labels = labels[isin]
    for i in range(labels.shape[-1]-1):
        transform_dict = align_labels(labels[:, i+1], labels[:, i])
        labs_rearranged = np.vectorize(transform_dict.get)(labels[:, i+1])
        labels[:, i+1] = labs_rearranged
    labels_new = np.full(isin.shape + labels.shape[-1:], -1)
    labels_new[isin] = labels
    return labels_new


def plot_labels_multi(labels, n_clusters_list, prefix):
    for i, n in enumerate(n_clusters_list):
        filename = f'{prefix}rearranged/nclusters{n:03d}/labels.png'
        plot_labels(labels[..., i], filename)


def main():

    prefix = sys.argv[1]  # e.g. 'data/her2st/H123/clusters-gene/'
    n_clusters_min = int(sys.argv[2])  # e.g. 6
    n_clusters_max = int(sys.argv[3])  # e.g. 14
    n_clusters_list = list(range(n_clusters_min, n_clusters_max+1))

    prefix_out = prefix + 'sankey/'

    labels = get_labels(prefix, n_clusters_list)
    labels = rearrange_labels_multi(labels)
    plot_labels_multi(labels, n_clusters_list, prefix_out)
    nodes_edges = get_multi_layers(labels)
    unify(nodes_edges)
    out = flatten(nodes_edges)
    plot_sankey(**out, filename=prefix_out+'sankey.png')


if __name__ == '__main__':
    main()
