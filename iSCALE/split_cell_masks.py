import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_pickle, save_pickle


def split_into_strata(data, x, cutoffs, prefix):
    start_list, stop_list = cutoffs[:-1], cutoffs[1:]
    stop_list[-1] += 1e-3
    for i, (start, stop) in enumerate(zip(start_list, stop_list)):
        isin = (x >= start) * (x < stop)
        indices = np.arange(len(x))[isin]
        data_strata = {}
        for ke, va in data.items():
            if isinstance(va, list):
                data_strata[ke] = [va[i] for i in indices]
            elif isinstance(va, pd.DataFrame):
                data_strata[ke] = va[isin]
            else:
                raise NotImplementedError()
        save_pickle(
                data_strata,
                f'{prefix}cell-level/area{i}/cell-data.pickle')


def plot_correlation(areas, counts, filename, xticks=None):
    counts = counts.sum(1).to_numpy()
    # counts[counts == 0] = 1
    # x = np.log2(areas)
    # y = np.log2(counts)
    x = areas
    y = counts
    corr = np.corrcoef(x, y)[0, 1]
    plt.scatter(x, y, alpha=0.1)
    plt.xlabel('cell area (um^2)')
    plt.ylabel('total gene expression count')
    if xticks is not None:
        plt.xticks(xticks)
    plt.title(f'correlation: {corr:.2f}')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(filename)


def get_cutoffs(x, stride=None, n_strata=None):
    if stride is not None:
        x_max = np.max(x)
        cutoffs = np.arange(0, x_max, stride)
        cutoffs = np.concatenate([cutoffs, [x_max]])

    elif n_strata is not None:
        qts = np.linspace(0, 1, n_strata+1)
        cutoffs = np.quantile(x, qts)

    return cutoffs


def main():

    prefix = sys.argv[1]
    # stride = 1000  # in um^2
    n_strata = 5

    data = load_pickle(prefix+'cell-level/cell-data.pickle')
    cutoffs = get_cutoffs(data['areas'], n_strata=n_strata)
    plot_correlation(
            data['areas'], data['counts'],
            filename='a.png')
    split_into_strata(data, data['areas'], cutoffs=cutoffs, prefix=prefix)


if __name__ == '__main__':
    main()
