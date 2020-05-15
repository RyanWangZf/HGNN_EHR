# -*- coding: utf-8 -*-
import os
import pdb

import numpy as np

from utils import read_symp2id, read_dise2id, load_embedding


if __name__ == '__main__':
    symp_emb, dise_emb = load_embedding("ckpt/gnn.pt",norm=True)
    dise2id, id2dise = read_dise2id()

    top_k = 5

    all_idxs = np.arange(1, dise_emb.shape[0])

    hard_dise_dict = {}

    for i in range(1,dise_emb.shape[0]):
        demb = dise_emb[i]
        all_left_idxs = np.setdiff1d(all_idxs, i)
        all_left_embs = dise_emb[all_left_idxs]

        sim = (demb * all_left_embs).sum(1) /  (np.linalg.norm(demb) * np.linalg.norm(all_left_embs, 2, 1))

        hard_d_id = all_left_idxs[np.argsort(-sim)][:top_k]

        hard_dise_dict[i] = hard_d_id

    np.save("dataset/hard_dise.npy", hard_dise_dict)
    print("Done")
    
