import sys

import numpy as np
from einops import reduce

from utils import load_image, load_tsv
from visual import plot_spots

inpfile_img = sys.argv[1]
inpfile_locs = sys.argv[2]
radius = int(sys.argv[3])
outfile = sys.argv[4]

im = load_image(inpfile_img)
if im.dtype == bool:
    im = im.astype(np.uint8) * 255
if im.ndim == 2:
    im = np.tile(im[..., np.newaxis], 3)

locs = load_tsv(inpfile_locs)
locs = locs[['y', 'x']]
locs = locs.astype(int)
locs = locs.to_numpy()

factor = 16
if factor is not None:
    im = reduce(
            im.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)
    locs //= factor
    radius //= factor

cnts = np.ones(locs.shape[0])
cnts[0] = 0

plot_spots(im, cnts, locs, radius, outfile, cmap='gray', weight=1.0)
