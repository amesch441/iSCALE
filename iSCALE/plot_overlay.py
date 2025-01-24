import sys

import numpy as np
import matplotlib
from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from visual import plot_spots
from utils import load_image, load_tsv, read_string
import math


prefix = sys.argv[1]  # e.g. data/xenium/rep0/
gene_name = sys.argv[2]  # e.g. CDH2
radius = sys.argv[3]  # e.g. a.jpg
outfile = sys.argv[4]  # e.g. a.jpg

cnts = load_tsv(prefix+'cnts.tsv')
locs = load_tsv(prefix+'locs.tsv')
#radius = read_string(prefix+'radius.txt')
radius = int(radius)

embs = get_embeddings(prefix)

#################################################################

#which locs are outside of the histology image? (we want to remove these spots)
x_out_low = locs.x < math.ceil(radius) 
x_out_high = locs.x > embs.shape[1]*16-math.ceil(radius)

y_out_low = locs.y < math.ceil(radius)
y_out_high = locs.y > embs.shape[0]*16-math.ceil(radius)


print(x_out_low)
print(y_out_low)
print(x_out_high)
print(y_out_high)


print(min(locs.x))
print(min(locs.y))
print(max(locs.x))
print(max(locs.y))

remove = x_out_high
for i in range(0,len(x_out_high)):
      remove[i] = x_out_low[i] or x_out_high[i] or y_out_low[i] or y_out_high[i]

print(sum(remove))

keep = ~remove
locs = locs[keep]
cnts = cnts[keep]

print(locs.shape)
print(cnts.shape)

#################################################################

assert all(cnts.index == locs.index)

cnts = cnts[gene_name].to_numpy()
cnts = np.log2(cnts+1)

locs = locs[['y', 'x']].astype(int).to_numpy()

img = load_image(prefix+'he.jpg')
img[:] = img.mean(-1, keepdims=True)

isin = np.logical_and(
        (locs >= radius).all(1),
        (locs < img.shape[:2]).all(1))
assert isin.all()

plot_spots(
        img=img, cnts=cnts, locs=locs, radius=radius,
        cmap='turbo', weight=0.8,
        outfile=outfile,
        boundary_color=matplotlib.colors.to_rgba('black')[:3])
