import sys

import numpy as np
from einops import reduce

from utils import load_pickle, save_pickle
from visual import plot_labels, plot_label_masks


def reduce_matrix(x, factor):
    return reduce(
            x, '(h1 h0) (w1 w0) -> h1 w1', 'mean',
            h0=factor, w0=factor)


def reduce_labels(lab, factor):
    lab_min = lab.min()
    lab = lab - lab_min
    onehot = [lab == la for la in range(lab.max()+1)]
    prob = [
            reduce_matrix(oh.astype(np.float32), factor)
            for oh in onehot]
    lab = np.argmax(prob, 0)
    lab = lab + lab_min
    return lab


def main():
    prefix = sys.argv[1]  # e.g. 'data/her2st/H1/clusters-gene/'
    factor = int(sys.argv[2])  # e.g. 4

    labels = load_pickle(prefix+'labels.pickle')

    labels = reduce_labels(labels, factor)

    outpref = f'{prefix}factor{factor:04d}/'
    save_pickle(labels, outpref+'labels.pickle')
    plot_labels(labels, outpref+'labels.png', cmap='tab10')
    plot_label_masks(labels, outpref+'masks/')


if __name__ == '__main__':
    main()
