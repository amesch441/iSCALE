import sys
import numpy as np
from utils import load_image, save_image
from image import crop_image
from einops import reduce


def stitch_images(imgs, axis=0, grid_shape=None, pad=0):

    shapes_raw = np.array([im.shape[:2] for im in imgs])

    if grid_shape is not None:
        filler = np.max([np.max(im) for im in imgs])
        empty = np.full_like(imgs[0][:1, :1], filler)
        n = len(imgs)
        imgs = imgs + [empty] * (grid_shape[0] * grid_shape[1] - n)
        imgs = [
                imgs[i*grid_shape[1]:(i+1)*grid_shape[1]]
                for i in range(grid_shape[0])]
        outs = [stitch_images(ims, axis=1-axis, pad=pad) for ims in imgs]
        imgs = [np.array(e[0]) for e in outs]
        origins_inner = [e[1] for e in outs]
        im, origins_outer, __ = stitch_images(imgs, axis=axis, pad=pad)
        origins = (
                np.expand_dims(origins_outer, axis=1-axis)
                + np.stack(origins_inner, axis=axis))
        origins = origins.reshape(-1, origins.shape[-1])
    else:
        shapes = shapes_raw.copy()
        shapes[:, 1-axis] = shapes[:, 1-axis].max()
        extents = [[(0, s+pad) for s in sha] for sha in shapes]
        imgs = [crop_image(im, ext) for im, ext in zip(imgs, extents)]
        im = np.concatenate(imgs, axis)
        starts = np.cumsum(np.concatenate([[0], shapes[:, axis]]))[:-1]
        origins = np.array([starts, starts]).T
        origins[:, 1-axis] = 0
    return im, origins, shapes_raw


def reduce_size(im, factor):
    if factor > 1:
        im = im.astype(np.float32)
        im = reduce(
                im, '(h1 h0) (w1 w0) c -> h1 w1 c', 'mean',
                h0=factor, w0=factor)
        im = im.astype(np.uint8)
    return im


def main():

    outfile = sys.argv[1]
    factor = int(sys.argv[2])
    inpfile_list = sys.argv[3:]

    imgs = [reduce_size(load_image(fname), factor) for fname in inpfile_list]
    im, __, __ = stitch_images(imgs, axis=1)
    save_image(im, outfile)


if __name__ == '__main__':
    main()
