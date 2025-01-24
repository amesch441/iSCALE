import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils import load_tsv, mkdir


def get_data_average(dataname, factor, metric):
    cnts_train_1, results_1 = get_data(dataname+'rep1/', factor, metric)
    cnts_train_2, results_2 = get_data(dataname+'rep2/', factor, metric)
    cnts_train = cnts_train_1 + cnts_train_2
    results = {
            ke: (results_1[ke]+results_2[ke])/2
            for ke in results_1.keys()}
    return cnts_train, results


def get_data(dataname, factor, metric):

    prefixs = {
            'XFuse': f'data/xfuse/{dataname}',
            'iStar': f'data/{dataname}',
            }
    cnts_train = load_tsv(prefixs['iStar']+'cnts.tsv')
    results = {
            ke: load_tsv(f'{pref}cnts-super-eval/factor{factor:04d}.tsv')
            for ke, pref in prefixs.items()}

    # match genes
    gene_names_list = [set(r.index.to_list()) for r in results.values()]
    gene_names_list.append(cnts_train.columns.to_list())
    gene_names = list(set.intersection(*gene_names_list))
    cnts_train = cnts_train[gene_names]
    results = {ke: r.loc[gene_names] for ke, r in results.items()}

    results = {ke: df[[metric]] for ke, df in results.items()}
    return cnts_train, results


def get_rank(df):
    rank = df.std().to_numpy().argsort().argsort()
    return rank


def match_genes(cnts_train, results):
    gene_names = cnts_train.columns.to_list()
    results = {
            ke: df.loc[gene_names] for ke, df in results.items()}
    results = {
            ke: df.to_numpy()[:, 0] for ke, df in results.items()}
    return results, gene_names


def plot_scatter(results, lim, outfile):
    # n_top = 50
    # is_top = rank >= rank.size - n_top
    # cmap = plt.get_cmap('tab10')
    # color = cmap(is_top)
    plt.figure(figsize=(4, 4))
    methods = list(results.keys())

    # # marginal box plots
    # plt.figure(figsize=(4, 4))
    # g = sns.JointGrid(
    #         data=results,
    #         x=methods[0],
    #         y=methods[1])
    # g.plot_joint(sns.scatterplot)
    # g.plot_marginals(sns.boxplot)

    # x = results[methods[0]]
    # y = results[methods[1]]
    # plt.scatter(x, y, alpha=0.5)

    sns.scatterplot(data=results, x=methods[0], y=methods[1])

    plt.axline(
            [lim[0], lim[0]], [lim[1], lim[1]],
            linestyle='--', color='tab:gray')
    # plt.axhline(0, color='black')
    # plt.axvline(0, color='black')
    # for gname, x, y in zip(
    #         gene_names, results[methods[0]], results[methods[1]]):
    #     plt.annotate(gname, xy=(x, y))
    plt.xlabel(methods[0])
    plt.ylabel(methods[1])
    plt.xlim(lim[0], lim[1])
    plt.ylim(lim[0], lim[1])

    mkdir(outfile)
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def plot_violins(results, rank, gene_names, lim, outfile):
    df = pd.DataFrame(results)
    df = df.add_prefix('correlation_')
    n_strata = 5
    quantile = rank / rank.size
    strata = (quantile * n_strata).astype(int)
    quantile = (strata + 0.5) / n_strata
    df['gene_variation_quantile'] = quantile
    df['gene'] = gene_names
    df = pd.wide_to_long(
            df, stubnames='correlation', i='gene', j='method',
            sep='_', suffix='\\D+')
    df = df.reset_index()

    plt.figure(figsize=(4, 4))
    sns.boxplot(
            data=df, x='gene_variation_quantile', y='correlation',
            hue='method', palette='muted', showfliers=False)
    plt.ylim(lim[0], lim[1])
    plt.xlabel('Gene Expression Variance')
    plt.ylabel('Prediction Correlation')
    plt.axhline(0.0, color='black')
    plt.legend(loc='upper left')
    plt.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close()
    print(outfile)


def main():
    dataname = sys.argv[1]
    factor = int(sys.argv[2])
    metric = sys.argv[3]

    average_results = dataname[-5:] not in ['rep1/', 'rep2/']

    if average_results:
        cnts_train, results = get_data_average(
                dataname, factor, metric=metric)
    else:
        cnts_train, results = get_data(dataname, factor, metric=metric)
    results, gene_names = match_genes(cnts_train, results)
    rank = get_rank(cnts_train)

    if factor >= 9000:
        pixsize = factor
    else:
        pixsize = 16 * factor

    if average_results:
        dataname += 'average/'

    outprefs = [
            f'data/{dataname}cnts-super-corr/',
            f'data/xfuse/{dataname}cnts-super-corr/']

    font = {'size': 15}
    plt.rc('font', **font)

    lim_dict = {
            'pearson': (-0.2, 1.0),
            'mpearson': (-0.2, 1.0),
            'spearman': (-0.2, 1.0),
            'rmse': (-0.05, 0.7),
            'psnr': (5, 30),
            'ssim': (-0.05, 1.05),
            }
    lim = lim_dict[metric]
    for pref in outprefs:
        plot_scatter(
                results, lim,
                outfile=f'{pref}{metric}/scatter-sidelen{pixsize:04d}.png')
        plot_violins(
                results, rank, gene_names, lim,
                outfile=f'{pref}{metric}/violins-sidelen{pixsize:04d}.png')


if __name__ == '__main__':
    main()
