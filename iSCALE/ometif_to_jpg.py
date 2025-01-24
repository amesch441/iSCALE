import sys

import skimage
import numpy as np

inpfile = sys.argv[1]  # e.g. 'rep2/outs/morphology_mip.ome.tif'
outfile = sys.argv[2]  # e.g. 'rep2/outs/morphology_mip.jpg'

qt = 0.01

im = skimage.io.imread(inpfile)
im = im - np.quantile(im, qt)
im = im / np.quantile(im, 1-qt) + 1e-12
im = np.clip(im, 0, 1)
im = (im * 255).astype(np.uint8)
skimage.io.imsave(outfile, im)
print(outfile)
