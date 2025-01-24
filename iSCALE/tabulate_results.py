import sys

import pandas as pd
import numpy as np

from utils import load_tsv

dataname = sys.argv[1]  # e.g. 'xenium/', 'xenium-mouse-brain/'

if dataname == 'xenium/':
    pref = 'data/'
    train_file = 'data/xenium/rep12/cnts.tsv'
    out_file = 'results/xenium/combined.csv'
    methods = {'iStar': '', 'XFuse': 'xfuse/'}
    tasks = {'In': '', 'Out': 'oos/'}
    sections = {'Sec1': 'rep1/', 'Sec2': 'rep2/'}
elif dataname == 'xenium-mouse-brain/':
    pref = 'data/'
    train_file = 'data/xenium-mouse-brain/cnts.tsv'
    out_file = 'results/xenium-mouse-brain/combined.csv'
    methods = {'iStar': '', 'XFuse': 'xfuse/'}
    tasks = {'In-sample': ''}
    sections = {'Section1': ''}
else:
    raise ValueError(f'Dataset name `{dataname}` not recognized.')

factors = {
        '128x': 1,
        '32x': 2,
        '8x': 4,
        '2x': 8,
        }
metrics = {
    'RMSE': 'rmse', 'SSIM': 'ssim', 'PCC': 'pearson'}

df = {}
for sec_name, sec in sections.items():
    for metr_name, met in metrics.items():
        for meth_name, meth in methods.items():
            for tas_name, tas in tasks.items():
                for fac_name, fac in factors.items():
                    inpfile = (
                            f'{pref}{meth}{dataname}{tas}{sec}'
                            f'cnts-super-eval/factor{fac:04d}.tsv')
                    x = load_tsv(inpfile)
                    x = x[met]
                    x.name = metr_name
                    x = (x * 100).round().astype(int)
                    x = x.astype(str).str.zfill(2)
                    key = (
                            f'{sec_name}-{metr_name}-{meth_name}-'
                            f'{tas_name}-{fac_name}')
                    df[key] = x

df = pd.DataFrame(df)

header = [tuple(s.split('-')) for s in df.columns]
header = pd.MultiIndex.from_tuples(header)
df.columns = header

cnts_train = load_tsv(train_file)
std = cnts_train.std(0)
mean = cnts_train.mean(0)
std = np.round(std, 1)
mean = np.round(mean, 1)
df.insert(0, 'SD', std)
df.insert(0, 'Mean', mean)
df = df.sort_values('SD', ascending=False)

df.index.name = 'Gene'

df.to_csv(out_file)
print(out_file)
