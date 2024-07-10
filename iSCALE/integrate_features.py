import numpy as np
from einops import rearrange

from reduce_dim import reduce_dim


def reduce_feature_dim(x, **kwargs):
    x = rearrange(x, 'c n -> 1 n c')
    x = reduce_dim(x, **kwargs)[0]
    x = rearrange(x, '1 n c -> c n')
    return x


def kronecker_product(xs):

    n = len(xs)
    dtype = xs[0].dtype

    y = np.array(1, dtype=dtype)
    for i, x in enumerate(xs):
        dims = [j for j in list(range(n)) if j != i]
        x = np.expand_dims(x, dims)
        y = y * x

    y = y.reshape((-1,) + y.shape[n:])

    return y


def integrate_features(
        xs, n_components_inp=None, n_components_out=None,
        include_original=False):

    shape = xs[0].shape[1:]

    if n_components_inp is None:
        n_components_inp = [0.95] * len(xs)

    xs = [x.reshape(x.shape[0], -1) for x in xs]
    xs_reduced = [
            reduce_feature_dim(x, n_components=nc)
            for x, nc in zip(xs, n_components_inp)]
    y = kronecker_product(xs_reduced)

    if n_components_out is not None:
        y = reduce_feature_dim(y, n_components=n_components_out)

    if include_original:
        y = np.concatenate([y, np.concatenate(xs_reduced)])

    y = y.reshape(y.shape[:1] + shape)

    return y
