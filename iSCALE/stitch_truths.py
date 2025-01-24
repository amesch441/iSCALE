import sys
import numpy as np

from stitch_images import stitch_images
from utils import load_pickle, save_pickle

outfile = sys.argv[1]
inpfile_list = sys.argv[2:]

truth_list = [load_pickle(inpfile) for inpfile in inpfile_list]
gene_names = np.array(truth_list[0]['gene_names'])
assert all([
    (gene_names == truth['gene_names']).all()
    for truth in truth_list])
cnts = stitch_images([truth['cnts'] for truth in truth_list])
data = dict(cnts=cnts, gene_names=gene_names)
save_pickle(data, outfile)
