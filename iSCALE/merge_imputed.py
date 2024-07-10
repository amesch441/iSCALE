import sys

import numpy as np
from einops import reduce

from utils import read_lines, load_pickle, save_pickle


def get_gene_counts(gene_names, prefix, factor=None):
    cnt_list = []
    for gn in gene_names:
        x = load_pickle(f'{prefix}cnts-super/{gn}.pickle')
        if factor is not None:
            x = reduce(
                    x, '(h1 h) (w1 w) -> h1 w1', 'sum', h=factor, w=factor)
        cnt_list.append(x)
    cnts = np.stack(cnt_list, -1)
    return cnts


def main():
    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    factor = int(sys.argv[2])  # e.g. 16
    gene_names = read_lines(f'{prefix}gene-names.txt')
    gene_names = np.array(gene_names)
    cnts = get_gene_counts(gene_names, prefix, factor=factor)
    save_pickle(
            dict(x=cnts, gene_names=gene_names),
            f'{prefix}cnts-super-merged/factor{factor:04d}.pickle')


if __name__ == '__main__':
    main()
