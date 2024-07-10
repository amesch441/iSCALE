import argparse
import numpy as np
from utils import load_pickle, load_mask
from visual import plot_cells


def aggregate_by_weights(x, indices_weights):
    out = [
            np.nansum(x[ind[:, 0], ind[:, 1]] * wt)
            for (ind, wt) in indices_weights]
    out = np.array(out)
    return out


def standardize(x):
    x = x - np.nanmin(x)
    x = x / (np.nanmax(x) + 1e-12)
    return x


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--gene', type=str, default=None)
    parser.add_argument('--embedding', type=str, default=None)
    parser.add_argument('--tissue', type=str, default=None)
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    # get cell masks and weights
    cell_data = load_pickle(args.cell)
    weights = cell_data['weights']
    interiors = cell_data['interiors']
    boundaries = cell_data['boundaries']

    # get values for cells
    if args.embedding is None:
        assert args.gene is not None
        x = cell_data['counts']
        x = x[args.gene].to_numpy()
    else:
        x = load_pickle(args.embedding)
        x = standardize(x)
        x = aggregate_by_weights(x, weights)
    x = standardize(x)

    # get tissue mask
    mask_tissue = None
    if args.tissue is not None:
        mask_tissue = load_mask(args.tissue)

    # visualize values based on cell masks
    boundaries = np.concatenate(boundaries).astype(int)
    plot_cells(
            x, interiors, boundaries=boundaries,
            tissue=mask_tissue, filename=args.out)


if __name__ == '__main__':
    main()
