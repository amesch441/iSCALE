import sys

import numpy as np
from einops import reduce
from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import load_image, load_tsv, read_lines, read_string
from visual import plot_spots
import matplotlib
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN

# def plot_spots(cnts, locs, underground, gene_names, radius, prefix):
#     under_weight = 0.2
#     cmap = plt.get_cmap('turbo')
#     under = underground.mean(-1, keepdims=True)
#     under = np.tile(under, 3)
#     under -= under.min()
#     under /= under.max() + 1e-12
#     for k, name in enumerate(gene_names):
#         x = cnts[:, k]
#         x = x - x.min()
#         x = x / (x.max() + 1e-12)
#         img = under * under_weight
#         for u, ij in zip(x, locs):
#             lower = np.clip(ij - radius, 0, None)
#             upper = np.clip(ij + radius, None, img.shape[:2])
#             color = np.array(cmap(u)[:3]) * (1 - under_weight)
#             img[lower[0]:upper[0], lower[1]:upper[1]] += color
#         img = (img * 255).astype(np.uint8)
#         save_image(img, f'{prefix}{name}.png')


def normalize(cnts):

    #embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    #embs_mean = np.nanmean(embs, (0, 1))
    #embs_std = np.nanstd(embs, (0, 1))
    #embs -= embs_mean
    #embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return cnts, (cnts_min, cnts_max)

def get_data_smooth_locs_cnts(prefix, dist):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]

    #normalize data to remove batch effects across daughter captures prior to smoothing
    cnts, (cnts_min, cnts_max) = normalize( cnts)

    # Extract coordinates as numpy array
    adata = pd.read_csv(f'{prefix}locs.tsv', sep = '\t')
    coords = np.array(adata.iloc[:,[1,2]])
    
    # Compute pairwise distances
    distance_matrix = squareform(pdist(coords, metric='euclidean'))

    # Perform clustering with a maximum distance of 100
    db = DBSCAN(eps=dist, min_samples=1, metric='precomputed')
    clusters = db.fit_predict(distance_matrix)
    locs = adata.iloc[:,[1,2]]

    # Add cluster labels to the locs DataFrame
    locs['cluster'] = clusters

    # Compute average locations for each cluster
    locs2 = locs.groupby('cluster').agg({'x': 'mean', 'y': 'mean'}).reset_index(drop=True)

    # Compute average gene expressions for each cluster
    cnts['cluster'] = clusters
    cnts2 = cnts.groupby('cluster').mean().reset_index(drop=True)

    print("Integrated matrix sizes (locs, cnts):")
    print(locs2.shape)
    print(cnts2.shape)

  # change xy coordinates to ij coordinates
    locs2 = np.stack([locs2['y'], locs2['x']], -1)
    target_shape=None 

    # match coordinates of embeddings and spot locations
    if target_shape is not None:
        wsi = load_image(f'{prefix}he.jpg')
        current_shape = np.array(wsi.shape[:2])
        rescale_factor = current_shape // target_shape
        locs2 = locs2.astype(float)
        locs2 /= rescale_factor

    # find the nearest pixel
    locs2 = locs2.round().astype(int)
    locs2 = pd.DataFrame(locs2)
    locs2.columns = ['y','x']
    
    cnts2 = pd.DataFrame(cnts2)
    cnts2.columns = list(gene_names)

    return locs2, cnts2

def plot_spots_multi(
        cnts, locs, gene_names, radius, img, prefix):
    for i, gname in enumerate(gene_names):
        ct = cnts[:, i]
        outfile = f'{prefix}{gname}.png'
        plot_spots(
                img=img, cnts=ct, locs=locs, radius=radius,
                cmap='turbo', weight=0.8, 
                outfile=outfile)


def main():
    prefix = sys.argv[1]  # e.g. 'data/her2st/B1/'
    dist = int(sys.argv[2])  # e.g. 'data/her2st/B1/'
    factor = 16

    print(dist)

    infile_cnts = f'{prefix}cnts.tsv'
    infile_locs = f'{prefix}locs.tsv'
    infile_img = f'{prefix}he-bw.jpg'
    infile_genes = f'{prefix}gene-names.txt'
    infile_radius = f'{prefix}radius.txt'

    # load data
    #cnts = load_tsv(infile_cnts)
    #locs = load_tsv(infile_locs)

    locs, cnts = get_data_smooth_locs_cnts(prefix, dist)

    #print(cnts)
    #print(locs)

    assert (cnts.index == locs.index).all()
    spot_radius = int(read_string(infile_radius))
    img = load_image(infile_img)

    if img.dtype == bool:
        img = img.astype(np.uint8) * 255
    if img.ndim == 2:
        img = np.tile(img[..., np.newaxis], 3)

    # select genes
    gene_names = read_lines(infile_genes)
    cnts = cnts[gene_names]
    cnts = cnts.to_numpy()

    # recale image
    locs = locs.astype(float)
    locs = np.stack([locs['y'], locs['x']], -1)
    locs /= factor
    locs = locs.round().astype(int)
    img = reduce(
            img.astype(float), '(h1 h) (w1 w) c -> h1 w1 c', 'mean',
            h=factor, w=factor).astype(np.uint8)

    # rescale spot
    spot_radius = np.round(spot_radius / factor).astype(int)

    # plot spot-level gene expression measurements
    plot_spots_multi(
            cnts=cnts,
            locs=locs, gene_names=gene_names,
            radius=spot_radius, 
            img=img, prefix=prefix+'spots-integrated/')


if __name__ == '__main__':
    main()
