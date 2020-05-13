# -*- coding: utf-8 -*-
"""Usage: python EHR_retrieval.py main --symp=1 --mode="sds" --top_k=10
"""
import os
import pdb

import torch
import numpy as np

from utils import read_symp2id

def load_embedding(ckptpath="ckpt/GNN.pt"):
    res = torch.load(ckptpath)
    symp_emb = res["symp_embeds.weight"].cpu().numpy()
    dise_emb = res["dise_embeds.weight"].cpu().numpy()
    # need normalized
    symp_norm = np.linalg.norm(symp_emb, 2, 1)
    dise_norm = np.linalg.norm(dise_emb, 2, 1)

    symp_emb[1:] = symp_emb[1:] * np.expand_dims(1/symp_norm[1:],1)
    dise_emb[1:] = dise_emb[1:] * np.expand_dims(1/dise_norm[1:],1)

    return symp_emb, dise_emb

class EHR_retrieval:
    def __init__(self, prefix="./dataset/EHR", mode="sds"):
        test_filepath = os.path.join(prefix,"test/data_moresymp.txt")

        assert mode in ["sus", "sds", "mix"]
        self.mode = mode

        # maps path
        if mode in ["sds","mix"]:
            self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy"),allow_pickle=True).item()
            self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy"),allow_pickle=True).item()

        if mode in ["sus","mix"]:
            self.user2symp = np.load(os.path.join(prefix,"user2symp.npy"),allow_pickle=True).item()
            self.symp2user = np.load(os.path.join(prefix,"symp2user.npy"),allow_pickle=True).item()

        # load embeddings
        self.symp_embs, self.dise_embs = load_embedding("ckpt/GNN.pt")
        self.num_symp = self.symp_embs.shape[0]
        pass

    def __call__(self, symp_idx, top_k=5):
        # input a symptom,
        # output alternative symptoms following the predefined mode.
        
        symp_emb = self.symp_embs[int(symp_idx)]
        norm_symp_emb = symp_emb / np.linalg.norm(symp_emb, 2)

        if self.mode == "sds":
            symps_2_hop = self.SDS_sampling(symp_idx)

        elif self.mode == "sus":
            symps_2_hop = self.SUS_sampling(symp_idx)

        else:
            # mix mode
            symps_2_hop_1 = self.SDS_sampling(symp_idx)
            symps_2_hop_2 = self.SUS_sampling(symp_idx)
            symps_2_hop = np.concatenate([symps_2_hop_1,symps_2_hop_2], 0)
            symps_2_hop = np.unique(symps_2_hop)

        # calculate cos similarity
        symps_2_hop_emb = self.symp_embs[symps_2_hop]
        scores = np.sum(symps_2_hop_emb * norm_symp_emb, 1)

        all_scores = np.zeros(self.num_symp) - 1e9
        all_scores[symps_2_hop] = scores
        ranks = np.argsort(-all_scores)
        # because the 1st one is the input itself
        return ranks[1:1+top_k]

    def SDS_sampling(self, symp_idx):
        # 1st-hop
        dise_1_hop = self.symp2dise[str(symp_idx)]
        symps_2_hop = []

        # 2nd-hop
        for dise in dise_1_hop:
            symp_2_hop = self.dise2symp[dise]
            symps_2_hop.append(symp_2_hop)

        symps_2_hop = np.unique(np.concatenate(symps_2_hop, 0)).astype(int)
        return symps_2_hop


    def SUS_sampling(self, symp_idx):
        # 1st-hop
        user_1_hop = self.symp2user[str(symp_idx)]
        symps_2_hop = []

        # 2nd-hop
        for user in user_1_hop:
            symp_2_hop = self.user2symp[user]
            symps_2_hop.append(symp_2_hop)
        
        symps_2_hop = np.unique(np.concatenate(symps_2_hop, 0)).astype(int)
        return symps_2_hop


def main(**kwargs):
    symp2id, id2symp = read_symp2id()

    param = {"symp":1, 
    "mode":"sds",
    "top_k":5,
    }

    param.update(kwargs)

    symp_wd = id2symp[str(param["symp"])]

    # init the retrieval sysmtem
    ehr_ret = EHR_retrieval(mode=param["mode"])

    # do retrieval
    rank_symp = ehr_ret(symp_idx=param["symp"], top_k=param["top_k"])

    # map to words
    wd_list = []
    for symp in rank_symp:
        wd_list.append(id2symp.get(str(symp)))


    print(symp_wd)
    print(wd_list)




    # pdb.set_trace()
    pass


if __name__ == '__main__':
    import fire
    fire.Fire()




