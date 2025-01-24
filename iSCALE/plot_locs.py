import sys
import numpy as np
from utils import load_image, save_image, load_tsv

inpfile_img = sys.argv[1]
inpfile_locs = sys.argv[2]
radius = int(sys.argv[3])
outfile = sys.argv[4]

im = load_image(inpfile_img)
if im.ndim == 2:
    im = np.tile(im[..., np.newaxis], 3)

locs = load_tsv(inpfile_locs)
locs = locs[['y', 'x']]
locs = locs.astype(int)
locs = locs.to_numpy()
offset = np.arange(-radius, radius+1)
offset = np.stack(np.meshgrid(offset, offset, indexing='ij'), -1)
offset = np.expand_dims(offset, -2)
locs = locs + offset
locs = locs.reshape(-1, locs.shape[-1])

color = [0, 255, 0]  # green
im[locs[:, 0], locs[:, 1]] = color

save_image(im, outfile)
