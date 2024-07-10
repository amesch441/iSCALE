import numpy as np
from einops import reduce, repeat

from utils import load_pickle, load_tsv, save_pickle
from visual import plot_matrix


def get_truth(prefix):
    truth = load_pickle(
            prefix+'cnts-truth-agg/radius0004-stride01-square/data.pickle')
    cnts, gene_names = truth['cnts'], truth['gene_names']
    cnts = cnts.astype(np.float32)
    # convert 8x8 px to 16x16 px
    cnts = reduce(
            cnts, '(h1 h) (w1 w) c -> h1 w1 c', 'sum',
            h=2, w=2)
    out = {g: c for c, g in zip(cnts.transpose(2, 0, 1), gene_names)}
    return out


def plot_multi_res(x, start, end, prefix, gene, factor_list):
    x = x[start[0]:end[0], start[1]:end[1]]
    for fac in factor_list:
        z = x
        if fac > 1:
            z = reduce(
                    z, '(h0 h1) (w0 w1) -> h0 w0', 'sum',
                    h1=fac, w1=fac)
            z = repeat(
                    z, 'h0 w0 -> (h0 h1) (w0 w1)',
                    h1=fac, w1=fac)
        z = repeat(
                z, 'h0 w0 -> (h0 h1) (w0 w1)',
                h1=16, w1=16)
        degree = 128 // fac**2
        prefix_degree = f'{prefix}{degree:03d}x/{gene}'
        save_pickle(z, prefix_degree+'.pickle')
        plot_matrix(z, prefix_degree+'.png')


def main():
    factor_list = [1, 2, 4, 8]
    prefix = 'data/xenium/rep1/'
    inpfile = 'params.txt'
    inputs = load_tsv(inpfile)

    truth = get_truth(prefix)

    for i, inp in inputs.iterrows():

        start = inp[['origin_row', 'origin_col']].to_numpy().astype(int)
        size = inp[['size_row', 'size_col']].to_numpy().astype(int)
        start -= start % 128
        size += 128 - size % 128
        start //= 16
        size //= 16
        end = start + size

        gene = inp['gene']

        cnts = truth[gene]
        cnts_pred = load_pickle(f'{prefix}cnts-super/{gene}.pickle')

        plot_multi_res(
                x=cnts, start=start, end=end,
                prefix=prefix+'cnts-multires/truth/',
                gene=gene, factor_list=factor_list)
        plot_multi_res(
                x=cnts_pred, start=start, end=end,
                prefix=prefix+'cnts-multires/pred/',
                gene=gene, factor_list=factor_list)


if __name__ == '__main__':
    main()
