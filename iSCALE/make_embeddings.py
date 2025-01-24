import sys

import numpy as np

from utils import load_image, load_pickle, read_lines, save_pickle


def main():

    prefix = sys.argv[1]  # e.g. 'data/xfuse/xenium-mouse-brain/'

    gene_names = read_lines(f'{prefix}gene-names.txt')
    mask = load_image(f'{prefix}mask-small.png') > 0
    cnts = [
            load_pickle(f'{prefix}cnts-super/{gn}.pickle')
            for gn in gene_names]
    cnts = np.stack(cnts, -1)
    cnts[~mask] = np.nan
    cnts -= np.nanmean(cnts, (0, 1))
    cnts /= np.nanstd(cnts, (0, 1)) + 1e-12
    cnts = cnts.transpose(2, 0, 1)
    save_pickle(dict(cls=cnts), f'{prefix}embeddings-gene-score.pickle')


if __name__ == '__main__':
    main()
