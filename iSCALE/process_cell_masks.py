import sys

import numpy as np

from utils import load_pickle, save_pickle, read_string, load_tsv


def rescale_indices(indices, scale):
    return [ind * scale for ind in indices]


def aggregate_indices(indices, factor):
    out = []
    for ind in indices:
        ind = (ind / factor).astype(int)
        uniq, count = np.unique(ind, axis=0, return_counts=True)
        weight = count / factor**2
        out.append((uniq, weight))
    return out


def main():

    prefix = sys.argv[1]

    factor = 16

    masks = load_pickle(prefix+'cell-level/cell-masks-raw.pickle')
    interiors = masks['interiors']
    boundaries = masks['boundaries']
    vertices = masks['vertices']
    counts = load_tsv(prefix+'cell-level/cell-counts.tsv')

    # match masks and counts by labels
    labs = np.array(list(interiors.keys()))
    interiors = [interiors[lab] for lab in labs]
    boundaries = [boundaries[lab] for lab in labs]
    vertices = [vertices[lab] for lab in labs]
    counts = counts.T[labs].T

    # find scale
    pixel_size_raw = float(read_string(prefix+'pixel-size-raw.txt'))
    pixel_size = float(read_string(prefix+'pixel-size.txt'))
    scale = pixel_size_raw / pixel_size

    # find cell areas in um
    areas = [m.shape[0] * pixel_size**2 for m in interiors]

    # rescale mask indices
    interiors = rescale_indices(interiors, scale=scale)
    boundaries = rescale_indices(boundaries, scale=scale)
    vertices = rescale_indices(vertices, scale=scale)

    # aggregate mask indices
    weights = aggregate_indices(interiors, factor=factor)

    data = dict(
            interiors=interiors, weights=weights,
            counts=counts, areas=areas, boundaries=boundaries,
            vertices=vertices)
    save_pickle(data, prefix+'cell-level/cell-data.pickle')


if __name__ == '__main__':
    main()
