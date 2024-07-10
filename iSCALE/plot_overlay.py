import sys

import numpy as np
import matplotlib

from visual import plot_spots
from utils import load_image, load_tsv, read_string


prefix = sys.argv[1]  # e.g. data/xenium/rep0/
gene_name = sys.argv[2]  # e.g. CDH2
outfile = sys.argv[3]  # e.g. a.jpg

cnts = load_tsv(prefix+'cnts.tsv')
locs = load_tsv(prefix+'locs-raw.tsv')
radius = read_string(prefix+'radius-raw.txt')
radius = int(radius)
assert all(cnts.index == locs.index)

cnts = cnts[gene_name].to_numpy()
cnts = np.log2(cnts+1)

locs = locs[['y', 'x']].astype(int).to_numpy()

img = load_image(prefix+'he-raw.jpg')
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
