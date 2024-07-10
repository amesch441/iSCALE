import argparse
import multiprocessing

import torch
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np

from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import read_lines, read_string, save_pickle
from image import get_disk_mask
from train import get_model as train_load_model
from reduce_dim import reduce_dim
from visual import plot_spot_masked_image


class FeedForward(nn.Module):

    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
            # TODO: change activation to LeakyRelu(0.01)
            activation = nn.LeakyReLU(0.1, inplace=True)
        self.activation = activation
        self.residual = residual

    def forward(self, x, indices=None):
        if indices is None:
            y = self.linear(x)
        else:
            weight = self.linear.weight[indices]
            bias = self.linear.bias[indices]
            y = nn.functional.linear(x, weight, bias)
        y = self.activation(y)
        if self.residual:
            y = y + x
        return y


class ELU(nn.Module):

    def __init__(self, alpha, beta):
        super().__init__()
        self.activation = nn.ELU(alpha=alpha, inplace=True)
        self.beta = beta

    def forward(self, x):
        return self.activation(x) + self.beta


class ForwardSumModel(pl.LightningModule):

    def __init__(self, lr, n_inp, n_out):
        super().__init__()
        self.lr = lr
        self.net_lat = nn.Sequential(
                FeedForward(n_inp, 256),
                FeedForward(256, 256),
                FeedForward(256, 256),
                FeedForward(256, 256))
        self.net_out = FeedForward(
                256, n_out,
                activation=ELU(alpha=0.01, beta=0.01))
        self.save_hyperparameters()

    def inp_to_lat(self, x):
        return self.net_lat.forward(x)

    def lat_to_out(self, x, indices=None):
        x = self.net_out.forward(x, indices)
        return x

    def forward(self, x, indices=None):
        x = self.inp_to_lat(x)
        x = self.lat_to_out(x, indices)
        return x

    def training_step(self, batch, batch_idx):
        x, y_mean = batch
        y_pred = self.forward(x)
        y_mean_pred = y_pred.mean(-2)
        # TODO: try l1 loss
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('rmse', mse**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer


class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)
        isin = np.isfinite(x).all((-1, -2))
        self.x = x[isin]
        self.y = y[isin]
        self.locs = locs[isin]
        self.size = x_all.shape[:2]
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def show(self, channel_x, channel_y, prefix):
        mask = self.mask
        size = self.size
        locs = self.locs
        xs = self.x
        ys = self.y

        plot_spot_masked_image(
                locs=locs, values=xs[:, :, channel_x], mask=mask, size=size,
                outfile=f'{prefix}x{channel_x:04d}.png')

        plot_spot_masked_image(
                locs=locs, values=ys[:, channel_y], mask=mask, size=size,
                outfile=f'{prefix}y{channel_y:04d}.png')


def get_disk(img, ij, radius):
    i, j = ij
    patch = img[i-radius:i+radius, j-radius:j+radius]
    disk_mask = get_disk_mask(radius)
    patch[~disk_mask] = 0.0
    return patch


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def add_coords(embs):
    coords = np.stack(np.meshgrid(
            np.linspace(-1, 1, embs.shape[0]),
            np.linspace(-1, 1, embs.shape[1]),
            indexing='ij'), -1)
    coords = coords.astype(embs.dtype)
    mask = np.isfinite(embs).all(-1)
    coords[~mask] = np.nan
    embs = np.concatenate([embs, coords], -1)
    return embs


def reduce_embeddings(embs):
    # cls features
    cls, __ = reduce_dim(embs[..., :192], 0.99)
    # sub features
    sub, __ = reduce_dim(embs[..., 192:-3], 0.90)
    rgb = embs[..., -3:]
    embs = np.concatenate([cls, sub, rgb], -1)
    return embs


def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = get_embeddings(prefix)
    # embs = embs[..., :192]  # use high-level features only
    # embs = reduce_embeddings(embs)
    locs = get_locs(prefix, target_shape=embs.shape[:2])
    # embs = add_coords(embs)
    return embs, cnts, locs


def get_model_kwargs(kwargs):
    return get_model(**kwargs)


def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr, device='cuda'):

    print('x:', x.shape, ', y:', y.shape)

    x = x.copy()

    dataset = SpotDataset(x, y, locs, radius)
    dataset.show(
            channel_x=0, channel_y=0,
            prefix=f'{prefix}training-data-plots/')
    model = train_load_model(
            model_class=ForwardSumModel,
            model_kwargs=dict(
                n_inp=x.shape[-1],
                n_out=y.shape[-1],
                lr=lr),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            device=device)
    model.eval()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model, dataset


def normalize(embs, cnts):

    embs = embs.copy()
    cnts = cnts.copy()

    # TODO: check if adjsut_weights in extract_features can be skipped
    embs_mean = np.nanmean(embs, (0, 1))
    embs_std = np.nanstd(embs, (0, 1))
    embs -= embs_mean
    embs /= embs_std + 1e-12

    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12

    return embs, cnts, (embs_mean, embs_std), (cnts_min, cnts_max)


def impute(
        embs, cnts, locs, radius, epochs, batch_size, prefix,
        n_states=1, device='cuda', n_jobs=1):

    names = cnts.columns
    cnts = cnts.to_numpy()
    cnts = cnts.astype(np.float32)

    embs, cnts, __, (cnts_min, cnts_max) = normalize(embs, cnts)

    kwargs_list = [
            dict(
                x=embs, y=cnts, locs=locs, radius=radius,
                batch_size=batch_size, epochs=epochs, lr=1e-4,
                prefix=f'{prefix}states/{i:02d}/', device=device)
            for i in range(n_states)]

    if n_jobs is None or n_jobs < 1:
        n_jobs = n_states
    if n_jobs == 1:
        out_list = [get_model_kwargs(kwargs) for kwargs in kwargs_list]
    else:
        with multiprocessing.Pool(processes=n_jobs) as pool:
            out_list = pool.map(get_model_kwargs, kwargs_list)

    model_list = [out[0] for out in out_list]
    dataset_list = [out[1] for out in out_list]
    mask_size = dataset_list[0].mask.sum()

    cnts_range = np.stack([cnts_min, cnts_max], -1)
    cnts_range /= mask_size

    save_pickle(
            dict(
                model=model_list,
                outcome_range=cnts_range,
                outcome_names=names),
            prefix+'predictor.pickle')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=None)  # e.g. 400
    parser.add_argument('--n-states', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-jobs', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    embs, cnts, locs = get_data(args.prefix)
    args = get_args()

    factor = 16
    radius = int(read_string(f'{args.prefix}radius.txt'))
    radius = np.round(radius / factor).astype(int)

    n_train = cnts.shape[0]
    batch_size = min(128, n_train//16)

    impute(
            embs=embs, cnts=cnts, locs=locs, radius=radius,
            epochs=args.epochs, batch_size=batch_size,
            n_states=args.n_states, prefix=args.prefix,
            device=args.device, n_jobs=args.n_jobs)


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
