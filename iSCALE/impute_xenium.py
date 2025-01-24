import argparse
import multiprocessing
import os.path

import torch
torch.set_float32_matmul_precision('medium')
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.optim import Adam
from torch import nn
import numpy as np
from tqdm import tqdm
from impute_by_basic import get_gene_counts, get_embeddings, get_locs
from utils import read_lines, read_string, save_pickle
from image import get_disk_mask
from train import get_model as train_load_model
# from reduce_dim import reduce_dim
from visual import plot_matrix, plot_spot_masked_image


class FeedForward(nn.Module):

    def __init__(
            self, n_inp, n_out,
            activation=None, residual=False):
        super().__init__()
        self.linear = nn.Linear(n_inp, n_out)
        if activation is None:
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

    def __init__(self, lr, n_inp, n_out, n_ref, true_covet, nn_idx_mat):
        super().__init__()
        self.true_covet = true_covet
        self.pred_covet = None
        self.nn_idx_mat = nn_idx_mat

        self.lr = lr
        self.n_ref = n_ref
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
        mse = ((y_mean_pred - y_mean)**2).mean()
        loss = mse
        self.log('rmse', mse**0.5, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)
        return optimizer

    def load_reference_weights(self, state_dict):
        assert self.n_ref == state_dict['net_out.linear.weight'].shape[0]
        rep_weight = self.net_out.linear.weight.clone()
        rep_bias = self.net_out.linear.bias.clone()
        rep_weight[:self.n_ref, ] = state_dict['net_out.linear.weight']
        rep_bias[:self.n_ref, ] = state_dict['net_out.linear.bias']
        rep_dict = state_dict.copy()
        rep_dict['net_out.linear.weight'] = rep_weight
        rep_dict['net_out.linear.bias'] = rep_bias
        self.load_state_dict(rep_dict, strict=False)
        assert (self.net_out.linear.weight[:self.n_ref,] == state_dict['net_out.linear.weight']).all()
        assert (self.net_out.linear.bias[:self.n_ref,] == state_dict['net_out.linear.bias']).all()

class SpotDataset(Dataset):

    def __init__(self, x_all, y, locs, radius):
        super().__init__()
        print("In SpotDataset: embs_train's shape is", x_all.shape)
        print("In SpotDataset: cnts_train's shape is", y.shape)
        print("In SpotDataset: locs_train's shape is", locs.shape)

        mask = get_disk_mask(radius)
        x = get_patches_flat(x_all, locs, mask)
        isin = np.isfinite(x).all((-1, -2)).flatten()

        x = x[isin]
        y = y[isin]
        locs = locs[isin]
        if x.shape.__len__() > 3:
            x = x.reshape([x.shape[0], np.prod(x.shape[1:-1]), x.shape[-1]])

        self.x = x
        self.y = y
        self.locs = locs
        self.radius = radius
        self.mask = mask

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        patch = img[
                s[0]+r[0][0]:s[0]+r[0][1],
                s[1]+r[1][0]:s[1]+r[1][1]]
        if mask.all():
            x = patch
        else:
            x = patch[mask]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def get_ref_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')
    cnts_ref = get_gene_counts(f'{prefix}ref-')
    embs_ref = get_embeddings(f'{prefix}ref-')

    ## for genes that will be referenced
    ref_gene_names = cnts_ref.columns
    ref_gene_names = set(gene_names).intersection(set(ref_gene_names))
    ref_gene_names = list(ref_gene_names)
    print("gene numbers that will be used for reference:", ref_gene_names.__len__())

    cnts_ref = cnts_ref[ref_gene_names]
    locs_ref = get_locs(f'{prefix}ref-', target_shape=embs_ref.shape[:2])

    print("end of getting ref data")
    return cnts_ref, embs_ref, locs_ref

def get_train_test_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')

    cnts_train = get_gene_counts(f'{prefix}train-')
    cnts_train = cnts_train[gene_names]
    embs_train = get_embeddings(f'{prefix}train-')
    locs_train = get_locs(f'{prefix}train-', target_shape=embs_train.shape[:2])

    embs_test = get_embeddings(f'{prefix}test-')
    return cnts_train, embs_train, locs_train, embs_test

def get_data(prefix):
    gene_names = read_lines(f'{prefix}gene-names.txt')

    cnts = get_gene_counts(prefix)
    cnts = cnts[gene_names]
    embs = get_embeddings(prefix)
    locs = get_locs(prefix, target_shape=embs.shape[:2])

    return cnts, embs, locs

def get_model_kwargs(kwargs):
    return get_model(**kwargs)

def get_model(
        x, y, locs, radius, prefix, batch_size, epochs, lr,
        load_saved=True, device='cuda', asreference=False, n_ref=0):
    print('x:', x.shape, ', y:', y.shape)
    x = x.copy()

    '''
    from sklearn.neighbors import NearestNeighbors
    knn = 200
    nbrs = NearestNeighbors(n_neighbors=knn+1, algorithm='ball_tree').fit(locs)
    distances, nn_idx_mat = nbrs.kneighbors(locs)
    print("now finding knn...")

    def get_covet(cnts, nn_idx):
        cov_lst = []
        g_mean = torch.tensor(cnts.mean(axis=0), dtype=torch.float32).to('cuda')
        x = torch.tensor(cnts, dtype=torch.float32).to('cuda')
        for i in tqdm(range(cnts.shape[0])):
            neighbors_idx = nn_idx[i][1:]  # Exclude the spot itself
            neighbors_cnts = x[neighbors_idx]
            centered_neighbors = neighbors_cnts - g_mean
            cov_mat = torch.matmul(centered_neighbors.T, centered_neighbors) / (neighbors_cnts.size(0) - 1)
            cov_lst.append(cov_mat.cpu())
        return torch.stack(cov_lst)
    print("now calculating true covet...")
    ## true_covet = torch.zeros(y.shape[0], 50, 50)
    '''
    nn_idx_mat = None
    true_covet = None

    dataset = SpotDataset(x, y, locs, radius)
    model = train_load_model(
            model_class=ForwardSumModel,
            model_kwargs=dict(
                n_inp=x.shape[-1],
                n_out=y.shape[-1],
                n_ref=n_ref,
                lr=lr,
                nn_idx_mat=nn_idx_mat,
                true_covet=true_covet
            ),
            dataset=dataset, prefix=prefix,
            batch_size=batch_size, epochs=epochs,
            load_saved=load_saved, device=device)
    model.eval()
    if asreference:
        torch.save(model.state_dict(), prefix + 'ref_model_weights.pth')
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


def show_results(x, names, prefix):
    for name in ['CD19', 'MS4A1', 'ERBB2', 'GNAS']:
        if name in names:
            idx = np.where(names == name)[0][0]
            plot_matrix(x[..., idx], prefix+name+'.png')


def predict_single_out(model, z, indices, names, y_range):
    z = torch.tensor(z, device=model.device)
    y = model.lat_to_out(z, indices=indices)
    y = y.cpu().detach().numpy()
    # y[y < 0.01] = 0.0
    # y[y > 1.0] = 1.0
    y *= y_range[:, 1] - y_range[:, 0]
    y += y_range[:, 0]
    return y


def predict_single_lat(model, x):
    x = torch.tensor(x, device=model.device)
    z = model.inp_to_lat(x)
    z = z.cpu().detach().numpy()
    return z


def predict(
        model_states, x_batches, name_list, y_range, prefix,
        device='cuda'):

    # states: different initial values for training
    # batches: subsets of observations
    # groups: subsets outcomes

    batch_size_outcome = 100

    model_states = [mod.to(device) for mod in model_states]

    # get features of second last layer
    z_states_batches = [
            [predict_single_lat(mod, x_bat) for mod in model_states]
            for x_bat in x_batches]
    z_point = np.concatenate([
        np.median(z_states, 0)
        for z_states in z_states_batches])
    z_dict = dict(cls=z_point.transpose(2, 0, 1))
    save_pickle(
            z_dict,
            prefix+'embeddings-gene.pickle')
    del z_point

    # predict and save y by batches in outcome dimension
    idx_list = np.arange(len(name_list))
    n_groups_outcome = len(idx_list) // batch_size_outcome + 1
    idx_groups = np.array_split(idx_list, n_groups_outcome)
    for idx_grp in idx_groups:
        name_grp = name_list[idx_grp]
        y_ran = y_range[idx_grp]
        y_grp = np.concatenate([
            np.median([
                predict_single_out(mod, z, idx_grp, name_grp, y_ran)
                for mod, z in zip(model_states, z_states)], 0)
            for z_states in z_states_batches])
        for i, name in enumerate(name_grp):
            save_pickle(y_grp[..., i], f'{prefix}cnts-super/{name}.pickle')


def impute(
        embs_train, cnts_train, locs_train, embs_test, radius,
        epochs, batch_size, prefix,
        n_states=1, load_saved=True, device='cuda', n_jobs=1, asreference=False, n_ref=0):

    names = cnts_train.columns
    cnts_train = cnts_train.to_numpy()
    cnts_train = cnts_train.astype(np.float32)

    __, cnts_train, __, (cnts_train_min, cnts_train_max) = normalize(embs_train, cnts_train)

    # mask = np.isfinite(embs).all(-1)
    # embs[~mask] = 0.0

    kwargs_list = [
            dict(
                x=embs_train, y=cnts_train, locs=locs_train, radius=radius,
                batch_size=batch_size, epochs=epochs, lr=5e-5,
                prefix=f'{prefix}states/{i:02d}/',
                load_saved=load_saved, device=device, asreference=asreference, n_ref=n_ref)
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

    if not asreference:
        # embs[~mask] = np.nan
        cnts_train_range = np.stack([cnts_train_min, cnts_train_max], -1)
        cnts_train_range /= mask_size

        batch_size_row = 50
        n_batches_row = embs_test.shape[0] // batch_size_row + 1
        embs_batches = np.array_split(embs_test, n_batches_row)
        del embs_test, embs_train

        predict(
            model_states=model_list, x_batches=embs_batches,
            name_list=names, y_range=cnts_train_range,
            prefix=prefix, device=device)

        # show_results(cnts_pred, names, prefix)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--epochs', type=int, default=None)  # e.g. 400
    parser.add_argument('--n-states', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--n-jobs', type=int, default=1)
    parser.add_argument('--load-saved', action='store_true')
    parser.add_argument('--out-of-sample', action='store_true')
    parser.add_argument('--reference', action='store_true')
    parser.add_argument('--ref-epochs', type=int, default=400)
    parser.add_argument('--ref-n-states', type=int, default=1)
    parser.add_argument('--ref-n-jobs', type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    factor = 16
    radius = int(read_string(f'{args.prefix}radius.txt'))
    print("radius before factoring is", radius)
    radius = radius / factor
    print("radius after factoring is", radius)

    if args.reference:
        model_all_exist = True
        for i in range(args.ref_n_jobs):
            file_path = f'{args.prefix}states/{i:02d}/ref_model_weights.pth'
            if not os.path.exists(file_path):
                model_all_exist = False
                print(f"File does not exist: {file_path}")
            else:
                print(f"File exists: {file_path}")

        radius_ref = int(read_string(f'{args.prefix}ref-radius.txt'))
        print("radius_ref before factoring is", radius_ref)
        radius_ref = radius_ref / factor
        print("radius_ref after factoring is", radius_ref)
        cnts_ref, embs_ref, locs_ref = get_ref_data(prefix=args.prefix)
        print("cnts_ref's shape is", cnts_ref.shape)
        print("embs_ref's shape is", embs_ref.shape)
        print("locs_ref's shape is", locs_ref.shape)

        n_train = cnts_ref.shape[0]
        batch_size = min(128, n_train // 16)

        if not model_all_exist:
            print("start referencing...")
            impute(embs_train=embs_ref, cnts_train=cnts_ref, locs_train=locs_ref,
                   embs_test=embs_ref,
                   radius=radius_ref,
                   epochs=args.ref_epochs, batch_size=batch_size,
                   n_states=args.ref_n_states, prefix=args.prefix,
                   load_saved=True,
                   device=args.device, n_jobs=args.ref_n_jobs,
                   asreference=True)
            print("reference training has completed...")
            print("parameters has been saved in .pth...")
            n_ref = cnts_ref.shape[1]
            del embs_ref, cnts_ref, locs_ref
        else:
            n_ref = cnts_ref.shape[1]
            del embs_ref, cnts_ref, locs_ref
            print("Ref models all exist, skipping reference steps..")
            print("n_ref is", n_ref)
    else:
        n_ref = 0
        print("n_ref is", n_ref)

    if args.out_of_sample:
        cnts_train, embs_train, locs_train, embs_test = get_train_test_data(args.prefix)
        print("cnts_train's shape is", cnts_train.shape)
        print("embs_train's shape is", embs_train.shape)
        print("locs_train's shape is", locs_train.shape)
        print("embs_test's shape is", embs_test.shape)
    else:
        cnts_train, embs_train, locs_train = get_data(args.prefix)
        print("cnts's shape is", cnts_train.shape)
        print("embs's shape is", embs_train.shape)
        print("locs's shape is", locs_train.shape)
        embs_test = embs_train
        print("embs_test's shape is", embs_test.shape)

    n_train = cnts_train.shape[0]
    batch_size = min(128, n_train // 16)
    print("n_ref is", n_ref)
    print("start imputing...")
    impute(
        embs_train=embs_train, cnts_train=cnts_train, locs_train=locs_train,
        embs_test=embs_test,
        radius=radius,
        epochs=args.epochs, batch_size=batch_size,
        n_states=args.n_states, prefix=args.prefix,
        load_saved=args.load_saved,
        device=args.device, n_jobs=args.n_jobs, n_ref=n_ref)

if __name__ == '__main__':
    # torch.multiprocessing.set_start_method('spawn')
    main()
