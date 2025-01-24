import sys
import warnings
from typing import Optional, Sequence

import cv2 as cv
import numpy as np
from PIL import Image
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_fill_holes

# Based on https://github.com/ludvb/xfuse


Image.MAX_IMAGE_PIXELS = None


def rescale(
    image: np.ndarray, scaling_factor: float, resample: int = Image.NEAREST
) -> np.ndarray:
    r'''
    Rescales image by a given `scaling_factor`

    :param image: Image array
    :param scaling_factor: Scaling factor
    :param resample: Resampling filter
    :returns: The rescaled image
    '''
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize(
        [round(x * scaling_factor) for x in image_pil.size], resample=resample,
    )
    return np.array(image_pil)


def resize(
    image: np.ndarray,
    target_shape: Sequence[int],
    resample: int = Image.NEAREST,
) -> np.ndarray:
    r'''
    Resizes image to a given `target_shape`

    :param image: Image array
    :param target_shape: Target shape
    :param resample: Resampling filter
    :returns: The rescaled image
    '''
    image_pil = Image.fromarray(image)
    image_pil = image_pil.resize(target_shape[::-1], resample=resample)
    return np.array(image_pil)


def remove_fg_elements(mask: np.ndarray, size_threshold: float):
    r'''Removes small foreground elements'''
    labels, _ = label(mask)
    labels_unique, label_counts = np.unique(labels, return_counts=True)
    small_labels = labels_unique[
        label_counts < size_threshold ** 2 * np.prod(mask.shape)
    ]
    mask[np.isin(labels, small_labels)] = False
    return mask


def compute_tissue_mask(
    image: np.ndarray,
    convergence_threshold: float = 0.0001,
    size_threshold: float = 0.01,
    initial_mask: Optional[np.ndarray] = None,
    max_iter: int = 100,
) -> np.ndarray:
    r'''
    Computes boolean mask indicating likely foreground elements in histology
    image.
    '''
    # pylint: disable=no-member
    # ^ pylint fails to identify cv.* members
    original_shape = image.shape[:2]
    scale_factor = 1000 / max(original_shape)

    image = rescale(image, scale_factor, resample=Image.NEAREST)
    image = cv.blur(image, (5, 5))

    if initial_mask is None:
        initial_mask = (
            cv.blur(cv.Canny(image, 100, 200), (20, 20)) > 0
        )
    else:
        initial_mask = rescale(
                initial_mask, scale_factor, resample=Image.NEAREST)

    initial_mask = binary_fill_holes(initial_mask)
    initial_mask = remove_fg_elements(initial_mask, 0.1)  # type: ignore
    mask = np.where(initial_mask, cv.GC_PR_FGD, cv.GC_PR_BGD)
    mask = mask.astype(np.uint8)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = bgd_model.copy()

    print('Computing tissue mask:')

    for i in range(max_iter):
        old_mask = mask.copy()
        try:
            cv.grabCut(
                image,
                mask,
                None,
                bgd_model,
                fgd_model,
                1,
                cv.GC_INIT_WITH_MASK,
            )
        except cv.error as cv_err:
            warnings.warn(f'Failed to mask tissue\n{str(cv_err).strip()}')
            mask = np.full_like(mask, cv.GC_PR_FGD)
            break
        prop_changed = (mask != old_mask).sum() / np.prod(mask.shape)
        print('  Iteration %2d Δ = %.2f%%', i, 100 * prop_changed)
        if prop_changed < convergence_threshold:
            break

    mask = np.isin(mask, [cv.GC_FGD, cv.GC_PR_FGD])
    mask = cleanup_mask(mask, size_threshold)

    mask = resize(mask, target_shape=original_shape, resample=Image.NEAREST)

    return mask


def cleanup_mask(mask: np.ndarray, size_threshold: float):
    r'''Removes small background and foreground elements'''
    mask = ~remove_fg_elements(~mask, size_threshold)
    mask = remove_fg_elements(mask, size_threshold)
    return mask


def main():
    infile, outfile = sys.argv[1:3]
    img = Image.open(infile)
    img = np.array(img)
    mask = compute_tissue_mask(img, size_threshold=0.1)
    img[~mask] = 0
    Image.fromarray(img).save(outfile)


if __name__ == '__main__':
    main()
