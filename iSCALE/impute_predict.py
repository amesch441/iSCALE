import argparse

import numpy as np
import torch

from impute_train import get_data
from visual import plot_matrix
from utils import load_pickle, save_pickle


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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('prefix', type=str)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    return args


def main():

    args = get_args()

    embs, __, __ = get_data(args.prefix)

    model_pack = load_pickle(args.prefix+'predictor.pickle')
    model = model_pack['model']
    cnts_range = model_pack['outcome_range']
    gene_names = model_pack['outcome_names']

    batch_size_row = 50
    n_batches_row = embs.shape[0] // batch_size_row + 1
    embs_batches = np.array_split(embs, n_batches_row)
    del embs

    predict(
            model_states=model, x_batches=embs_batches,
            name_list=gene_names, y_range=cnts_range,
            prefix=args.prefix, device=args.device)
    # show_results(cnts_pred, names, prefix)


if __name__ == '__main__':
    main()
