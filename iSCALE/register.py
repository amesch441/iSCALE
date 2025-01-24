import sys
import pathlib
import os
from itertools import product

from valis import registration
from valis.registration import Valis
import numpy as np

from utils import load_tsv, save_tsv, load_pickle, save_pickle


def get_grid(shape):
    locs = np.meshgrid(
            np.arange(shape[0]), np.arange(shape[1]),
            indexing='ij')
    locs = np.stack(locs, axis=-1)
    return locs


def inverse_map(locs, shape):
    locs = np.round(locs).astype(int)
    locs_inv = np.full(tuple(shape)+(2,), -1)
    for i, j in product(range(locs.shape[0]), range(locs.shape[1])):
        locs_inv[tuple(locs[i, j])] = (i, j)
    return locs_inv


def get_slide_paths(prefix):
    dirname = os.path.dirname(prefix)
    basenames = os.listdir(dirname)
    basenames.sort()
    labels = [name.split('.')[0] for name in basenames]
    fullpaths = [os.path.join(dirname, name) for name in basenames]
    return {path: lab for path, lab in zip(fullpaths, labels)}


def register(dir_slides, dir_results, micro=False):

    # Create a Valis object and use it to register the slides in slide_src_dir,
    # aligning towards the reference slide.

    slide_paths = get_slide_paths(dir_slides)
    print(slide_paths)

    registrar = Valis(
            src_dir=os.path.dirname(dir_slides),
            dst_dir=os.path.dirname(dir_results),
            imgs_ordered=True,
            img_list=slide_paths)
    # Outputs: registrar_rigid, registrar_nonrigid, error_df
    registrar.register()

    if micro:
        # Perform micro-registration on higher resolution images
        # aligning directly to the reference image
        registrar_nonrigid, error_df_micro = registrar.register_micro(
                max_non_rigid_registartion_dim_px=2000,
                align_to_reference=True)

    return registrar


def main():

    prefix = sys.argv[1]  # e.g. 'data/registration/G/'

    save_maps = False

    dir_slides = prefix + 'slides/'
    dir_locs = prefix + 'locs/'
    dir_results = prefix + 'results/'
    dir_outputs = prefix + 'outputs/'
    model_path = prefix + 'registrar.pickle'

    if os.path.exists(model_path):
        registrar = load_pickle(model_path)
    else:
        registrar = register(dir_slides=dir_slides, dir_results=dir_results)
        save_pickle(registrar, model_path)

    # Save all registered slides as ome.tiff
    registrar.warp_and_save_slides(
            dir_outputs+'slides', crop='overlap')

    if save_maps:
        # save forard and backward registration maps
        forward, backward = {}, {}
        for name, slide in registrar.slide_dict.items():

            shape_src = tuple(slide.image.shape[:2])
            shape_tar = tuple(slide.aligned_slide_shape_rc)

            locs_src = get_grid(shape_src)
            locs_src = locs_src.reshape(-1, locs_src.shape[-1])
            locs_tar = slide.warp_xy(locs_src[:, ::-1])[:, ::-1]
            locs_tar = locs_tar.reshape(shape_src + locs_tar.shape[-1:])

            locs_inv = inverse_map(locs_tar, shape_tar)
            forward[name] = locs_tar
            backward[name] = locs_inv
        save_pickle(
                dict(forward=forward, backward=backward),
                f'{dir_outputs}transformation.pickle')

    # transform customized locations
    locs_file_list = list(pathlib.Path(dir_locs).rglob('*.tsv'))
    for filename in locs_file_list:
        dirname, basename = os.path.split(filename)
        stemname = basename.split('.tsv')[0]
        slide = registrar.get_slide(stemname)
        locs_df = load_tsv(filename)
        locs = locs_df[['x', 'y']].to_numpy()
        locs = slide.warp_xy(locs)
        locs = np.round(locs).astype(int)
        locs_df[:] = locs
        save_tsv(locs_df, f'{dir_outputs}locs/{stemname}.tsv')

    if False:
        # Kill the JVM
        registration.kill_jvm()


if __name__ == '__main__':
    main()
