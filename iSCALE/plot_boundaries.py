import argparse

import numpy as np
from matplotlib import colors

from utils import load_image, load_tsv, read_string, save_image, load_pickle
from image import get_disk_mask, upscale


def to_rgb(name):
    color = colors.to_rgba(name)
    color = (np.array(color[:3]) * 255).astype(int)
    return color


def get_args():

    parser = argparse.ArgumentParser()
    # Image file (e.g. xenium/rep1/cnts-super-plots/ERBB2.png)
    parser.add_argument('--img', type=str)
    # Location file (e.g. xenium/rep1/locs.tsv)
    parser.add_argument('--locs', type=str)
    # Radius file (e.g. xenium/rep1/radius.tsv)
    parser.add_argument('--radius', type=str)
    # Radius file (e.g. xenium/rep1/radius.tsv)
    parser.add_argument('--cell', type=str)
    # Output file (e.g. xenium/rep1/cell-level/cell-data.pickle)
    parser.add_argument('--out', type=str)
    # Upscaling factor (e.g. 16)
    parser.add_argument('--factor', type=int, default=None)
    # Spot boundary color name (e.g. black)
    parser.add_argument('--color', type=str, default='black')
    args = parser.parse_args()
    return args


def get_spot_masks(locs, radius, img_shape):

    locs = locs[['y', 'x']]
    locs = locs.astype(int)
    locs = locs.to_numpy()

    mask_patch = get_disk_mask(radius, boundary_width=radius//10)
    mask = np.zeros(img_shape).astype(bool)
    for i, j in locs:
        mask[i-radius:i+radius, j-radius:j+radius] = mask_patch
    return mask


def main():

    args = get_args()
    img = load_image(args.img)
    color = to_rgb(args.color)

    if args.factor is not None:
        target_shape = np.array(img.shape[:2]) * args.factor
        img = upscale(img, target_shape=target_shape)

    if (args.locs is not None) and (args.radius is not None):
        locs = load_tsv(args.locs)
        radius = int(read_string(args.radius))
        mask = get_spot_masks(locs, radius, img.shape[:2])
        img[mask] = color
    elif args.cell is not None:
        cell_data = load_pickle(args.cell)
        boundaries = cell_data['boundaries']
        boundaries = np.concatenate(boundaries).astype(int)
        img[boundaries[:, 0], boundaries[:, 1]] = color

    save_image(img, args.out)


if __name__ == '__main__':
    main()
