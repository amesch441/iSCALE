import sys

import numpy as np
import matplotlib.pyplot as plt

from utils import load_pickle, load_tsv, read_string, save_image
from image import get_disk_mask, upscale


prefix = sys.argv[1]
gene_name = sys.argv[2]
outfile = sys.argv[3]

factor = 16
cmap = plt.get_cmap('turbo')

x = load_pickle(f'{prefix}cnts-super/{gene_name}.pickle')
locs = load_tsv(f'{prefix}locs.tsv')
radius = int(read_string(f'{prefix}radius.txt'))

if factor > 1:
    x = upscale(x, target_shape=np.array(x.shape[:2])*factor)

locs = locs[['y', 'x']]
locs = locs.astype(int)
locs = locs.to_numpy()

locs = (locs / 16 * factor).astype(int)
radius = int(radius / 16 * factor)

mask_foregound = np.isfinite(x)

mask_patch = get_disk_mask(radius)
mask_spots = np.zeros(x.shape[:2]).astype(bool)
for i, j in locs:
    mask_spots[i-radius:i+radius, j-radius:j+radius] = mask_patch

x[~mask_spots] = np.nan
x -= np.nanmin(x)
x /= np.nanmax(x) + 1e-12
x = cmap(x)[..., :3]
x[~mask_foregound] = 1  # set pixels outside tissue to white
x[(~mask_spots) * mask_foregound] = 0  # set non-spots inside tissue to black
x = (x * 255).astype(np.uint8)
save_image(x, outfile)
