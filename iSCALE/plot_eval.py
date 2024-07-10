import sys

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from utils import load_tsv

prefix = sys.argv[1]

infile = prefix + '.tsv'
outfile = prefix + '.png'

df = load_tsv(infile)
df = df.reset_index()

df = pd.wide_to_long(df, stubnames='correlation', i='gene', j='pixel_size')
df = df.reset_index()

plt.figure(figsize=(8, 8))
ax = sns.violinplot(
        data=df, x='pixel_size', y='correlation',
        color='lightgray', saturation=0.5)
sns.boxplot(
        data=df, x='pixel_size', y='correlation',
        color='lightblue', saturation=0.5, width=0.2,
        boxprops={'zorder': 2}, ax=ax)
plt.ylim(-0.1, 1.0)
plt.axhline(0.0, color='black', linewidth=0.5)
plt.title('Imputation performance')
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()
print(outfile)
