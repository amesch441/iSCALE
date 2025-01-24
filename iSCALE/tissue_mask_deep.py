import argparse

import numpy as np
from scipy.spatial.distance import dice as dice_distance
from scipy.ndimage import binary_fill_holes
from skimage.transform import resize as resize_image
from einops import reduce

from extract_features import get_embeddings_shift as get_embs
from image import smoothen
from cluster import cluster
from tissue_mask import compute_tissue_mask


def resize(img, shape):
    return resize_image(img, shape, order=3, preserve_range=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args


def process_image(img):
    mask = np.full(img[..., [0]].shape, 255, dtype=np.uint8)
    img = np.concatenate([img, mask], -1)
    return img


def get_embeddings(img, device='cuda'):
    x_cls, x_sub = get_embs(
            img, stride=128, pretrained=True, device=device)
    x_cls = smoothen(np.stack(x_cls, -1), size=16, method='cv')
    x_sub = smoothen(np.stack(x_sub, -1), size=16, method='cv')
    x = np.concatenate([x_cls, x_sub], -1)
    return x


def get_labels(x):
    labels, __ = cluster(
            x.transpose(2, 0, 1), n_clusters=2, method='km')
    return labels


def get_mask_init(img):
    mask = compute_tissue_mask(
            img[..., :3], size_threshold=0.1, initial_mask=None,
            max_iter=10)
    factor = 16
    mask = reduce(
            mask.astype(np.float32),
            '(h0 h1) (w0 w1) -> h0 w0', 'mean',
            h1=factor, w1=factor) > 0.5
    return mask


def select_label(labels, mask):
    dist_best = np.inf
    lab_best = None
    for lab in np.unique(labels):
        ma = labels == lab
        dist = dice_distance(ma.flatten(), mask.flatten())
        if dist < dist_best:
            dist_best = dist
            lab_best = lab
    return labels == lab_best


def upscale_mask(mask):
    factor = 16
    shape = np.array(mask.shape) * factor
    mask = mask.astype(np.float32)[..., np.newaxis]
    mask = resize(mask, shape)
    mask = mask[..., 0] > 0.5
    return mask


def compute_foreground(img):
    img = process_image(img)
    x = get_embeddings(img, device='cuda')
    labels = get_labels(x)
    mask_init = get_mask_init(img)
    mask = select_label(labels, mask_init)
    mask = binary_fill_holes(mask)
    mask = upscale_mask(mask)
    return mask
