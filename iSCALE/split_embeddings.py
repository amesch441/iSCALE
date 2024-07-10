import numpy as np
from utils import load_pickle, save_pickle


def crop_embs(embs, ext):
    return {
            ke: [
                c[ext[0][0]:ext[1][0], ext[0][1]:ext[1][1]]
                for c in channels]
            for ke, channels in embs.items()}


def main():

    infile = 'data/xenium/rep12/embeddings-hist.pickle'

    extent_list = [
            ((0, 0), (15104, 20224)),
            ((0, 20480), (13824+0, 20480+20480)),
            ]
    extent_list = np.array(extent_list) // 16

    embs = load_pickle(infile)

    embs1, embs2 = [crop_embs(embs, extent) for extent in extent_list]
    save_pickle(
            embs1, 'data/xenium/rep12/rep1/embeddings-hist.pickle')
    save_pickle(
            embs2, 'data/xenium/rep12/rep2/embeddings-hist.pickle')


if __name__ == '__main__':
    main()
