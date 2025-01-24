import sys
import os
import numpy as np
import pandas as pd
import pickle
import argparse
import matplotlib.pyplot as plt
from utils import save_image, save_pickle, load_pickle, load_mask
import harmonypy as hm

#put all (hipt or other ).pickles to read after --embs, and put all .pickles' name to write out #after --output
#like:
#python harmony.py --embs ../xenium/CRC-P1-embeddings-hipt-raw.pickle ../xenium/CRC-P2-#embeddings-hipt-raw.pickle ../xenium/CRC-P5-embeddings-hipt-raw.pickle --output ../xenium/CRC-#P1-embeddings-hipt-raw-har.pickle ../xenium/CRC-P2-embeddings-hipt-raw-har.pickle ../xenium/#CRC-P5-embeddings-hipt-raw-har.pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--embs",
        nargs="+",
        type=str,
        help="A list of embedding pickles to be harmony-ed "
    )
    parser.add_argument(
        "--output",
        nargs="+",
        type=str,
        help="A list of embedding pickles after harmony-ed "
    )
    args = parser.parse_args()
    return args

def main():
    embs_list = []
    args = get_args()
    dir_list = args.embs
    print(dir_list)
    for d in dir_list:
        x = load_pickle(d)
        if isinstance(x, dict):
            x = np.concatenate([x['cls'], x['sub'], x['rgb']])
            x = x.transpose(1, 2, 0)
        embs_list.append(x)
    meta_data = pd.DataFrame(columns=['batch', 'row', 'col'])
    df_list = []
    ary_list = []
    for i in range(dir_list.__len__()):
        x = embs_list[i]
        d = dir_list[i]
        df = pd.DataFrame({'batch': [d] * (x.shape[0] * x.shape[1]),
                           'row': np.meshgrid(range(x.shape[0]), range(x.shape[1]), indexing='ij')[0].flatten(),
                           'col': np.meshgrid(range(x.shape[0]), range(x.shape[1]), indexing='ij')[1].flatten()})
        print(df.shape)
        df_list.append(df)
        ary_list.append(x.reshape(x.shape[0] * x.shape[1], x.shape[2]))
    meta_data = pd.concat(df_list)
    print(meta_data.shape)
    vars_use = ['batch']
    data_mat = np.vstack(ary_list)
    assert meta_data.shape[0] == data_mat.shape[0]
    print(data_mat.shape)

    data_mat = data_mat.astype(np.float32)
    print("Running harmony now")
    ho = hm.run_harmony(data_mat, meta_data, vars_use)
    print("Done running harmony")
    h = np.array(ho.Z_corr).T

    output_list = args.output
    assert output_list.__len__() == dir_list.__len__()

    idx = [x.shape[0]*x.shape[1] for x in embs_list]
    cum_idx = np.cumsum([0] + idx)
    out = [h[cum_idx[i]:cum_idx[i + 1]] for i in range(len(idx))]
    for i in range(dir_list.__len__()):
        x = embs_list[i]
        save_pickle(out[i].reshape(x.shape[0], x.shape[1], x.shape[2]), output_list[i])

if __name__ == '__main__':
    main()
