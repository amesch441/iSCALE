import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import load_tsv

n_clusters = 10
change_threshold = 2.0
fp_threshold = np.inf

pref = sys.argv[1]

pref = (
        pref
        + 'cnts-clustered/by-clusters/'
        + 'contrast/by-clusters/')
suff = '.tsv'

dfs = [load_tsv(f'{pref}cluster-{i}{suff}') for i in range(n_clusters)]

fp_max = max([df['mean_exterior'].max() for df in dfs])
# change_max = max([df['fold_change'].max() for df in dfs])

plt.figure(figsize=(32, 16))
markers = []
for i, df in enumerate(dfs):
    markers.append([])
    gene_names = df.index
    fp = df['mean_exterior'].to_numpy()
    change = df['fold_change'].to_numpy()
    plt.subplot(2, 5, i+1)
    plt.plot(fp, change, 'o')

    is_specific = fp < fp_threshold
    is_changed = change > max(change_threshold, np.quantile(change, 0.95))
    is_marker = np.logical_and(is_specific, is_changed)
    indices = np.arange(is_marker.size)[is_marker]
    indices = indices[change[indices].argsort()[::-1]]
    for idx in indices:
        markers[-1].append(gene_names[idx])
        plt.annotate(gene_names[idx], (fp[idx], change[idx]))
    plt.xlim(0, fp_max * 1.1)
    # plt.ylim(0, change_max * 1.1)
    plt.xlabel('exterior mean')
    plt.ylabel('fold change')
    plt.title(f'Cluster {i}')
outfile = pref+'differential-top.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()
print(outfile)

overlap = np.full((len(markers),)*2, np.nan)
for i in range(len(markers)):
    for j in range(i):
        comm = set(markers[i]).intersection(markers[j])
        base = (len(markers[i]) + len(markers[j])) / 2
        olap = len(comm) / max(5, base)
        overlap[i, j] = olap
plt.xlabel('cluster')
plt.ylabel('cluster')
plt.title('Overlap between marker genes')
plt.imshow(overlap)
plt.colorbar()
outfile = pref+'differential-similarity.png'
plt.savefig(outfile, dpi=300, bbox_inches='tight')
plt.close()
print(outfile)
