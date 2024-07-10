import sys

import numpy as np
from einops import reduce

from utils import read_lines, load_pickle, load_image
from visual import plot_matrix
from scipy import stats




def replace_with_percentile(matrix):
    # Find the percentiles of non-NaN values
    non_nan_values = matrix[~np.isnan(matrix)]
    percentiles = np.percentile(non_nan_values, np.linspace(0, 100, num=100))
    # Replace non-NaN values with their percentiles
    replaced_matrix = np.copy(matrix)
    non_nan_indices = ~np.isnan(matrix)
    values = matrix[non_nan_indices]
    percentile_indices = np.searchsorted(percentiles, values)
    replaced_matrix[non_nan_indices] = percentile_indices / 100.0
    return replaced_matrix


def compute_score_percentile(cnts, mask=None, factor=None):
    if mask is not None:
        cnts[~mask] = np.nan

    ## Compute percentile for each gene
    for i in range(cnts.shape[2]):
        print(i)
        cnts[:,:,i] = replace_with_percentile(cnts[:,:,i])


    if factor is not None:
        cnts = reduce(
                cnts, '(h0 h1) (w0 w1) c -> h0 w0 c', 'mean',
                h1=factor, w1=factor)

    cnts -= np.nanmin(cnts, (0, 1))
    cnts /= np.nanmax(cnts, (0, 1)) + 1e-12


    score = cnts.mean(-1)

    return score




def compute_score(cnts, mask=None, factor=None):
    if mask is not None:
        #cnts = cnts.flatten()
        #print(mask)
        #print(cnts)


        mask = mask[:,:,0] #here
        cnts[~mask] = np.nan

    if factor is not None:
        cnts = reduce(
                cnts, '(h0 h1) (w0 w1) c -> h0 w0 c', 'mean',
                h1=factor, w1=factor)

    cnts -= np.nanmin(cnts, (0, 1))
    cnts /= np.nanmax(cnts, (0, 1)) + 1e-12


    score = cnts.mean(-1)

    return score





def get_marker_score(prefix, genes_marker, factor=1):

    genes = read_lines(prefix+'gene-names.txt')
    mask = load_image(prefix+'mask-small.png') > 0

    gene_names = set(genes_marker).intersection(genes)
    cnts = [
            load_pickle(f'{prefix}cnts-super/{gname}.pickle')
            for gname in gene_names]


    #cnts = np.stack(cnts, -1, dtype='float32')
    cnts = np.stack(cnts, -1)
    score = compute_score(cnts, mask=mask, factor=factor)
    return score

def get_marker_score_percentile(prefix, genes_marker, factor=1):

    genes = read_lines(prefix+'gene-names.txt')
    mask = load_image(prefix+'mask-small.png') > 0

    gene_names = set(genes_marker).intersection(genes)
    cnts = [
            load_pickle(f'{prefix}cnts-super/{gname}.pickle')
            for gname in gene_names]


    #cnts = np.stack(cnts, -1, dtype='float32')
    cnts = np.stack(cnts, -1)
    score = compute_score_percentile(cnts, mask=mask, factor=factor)
    return score


def main():

    prefix = sys.argv[1]  # e.g. 'data/her2st/H123/'
    genes_marker_file = sys.argv[2]  # e.g. 'data/markers/tls.txt'
    outfile = sys.argv[3]  # e.g. 'data/her2st/H123/tls.png'

    # compute marker score
    genes_marker = read_lines(genes_marker_file)
    score = get_marker_score(prefix, genes_marker)

    # visualize marker score
    score = np.clip(
            score, np.nanquantile(score, 0.001),
            np.nanquantile(score, 0.999))
    plot_matrix(score, outfile, white_background=True)


if __name__ == '__main__':
    main()
