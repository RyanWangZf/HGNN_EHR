# -*- coding: utf-8 -*-
"""Usage: python EHR_retrieval.py main --symp=1 --mode="sds" --top_k=10
"""
import os
import pdb

import torch
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_symp2id, load_embedding

from scipy import sparse

class EHR_retrieval:
    def __init__(self, prefix="./dataset/EHR", mode="sds"):
        test_filepath = os.path.join(prefix,"test/data_moresymp.txt")

        assert mode in ["sus", "sds", "mix", "pmi"]
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

        if mode in ["pmi"]:
            # init a PMI matrix that has shape M X M
            # we'd better make it a sparse matrix
            # read data
            self.pmi_mat_path = os.path.join(prefix, "pmi_mat.npz")
            self.symp_count_path = os.path.join(prefix, "sympcount.npy")
            if os.path.exists(self.pmi_mat_path):
                print("Load PMI Mat from", self.pmi_mat_path)
                self.symp2symp = sparse.load_npz(self.pmi_mat_path)
                self.sympcount = np.load(self.symp_count_path, allow_pickle=True).item()
                self.symp2symp.setdiag(0)
            else:
                print("Build PMI Mat.")
                self.build_pmi_matrix(prefix)
                self.symp2symp.setdiag(0)


    def __call__(self, symp_idx, top_k=5):
        # input a symptom,
        # output alternative symptoms following the predefined mode.
        
        symp_emb = self.symp_embs[int(symp_idx)]
        norm_symp_emb = symp_emb / np.linalg.norm(symp_emb, 2)

        if self.mode == "sds":
            symps_2_hop = self.SDS_sampling(symp_idx)

        if self.mode == "sus":
            symps_2_hop = self.SUS_sampling(symp_idx)

        if self.mode == "mix":
            # mix mode
            symps_2_hop_1 = self.SDS_sampling(symp_idx)
            symps_2_hop_2 = self.SUS_sampling(symp_idx)
            symps_2_hop = np.concatenate([symps_2_hop_1,symps_2_hop_2], 0)
            symps_2_hop = np.unique(symps_2_hop)

        if self.mode in ["sds","sus","mix"]:
            # calculate cos similarity
            symps_2_hop_emb = self.symp_embs[symps_2_hop]
            scores = np.sum(symps_2_hop_emb * norm_symp_emb, 1)

            all_scores = np.zeros(self.num_symp) - 1e9
            all_scores[symps_2_hop] = scores
            ranks = np.argsort(-all_scores)
            # because the 1st one is the input itself
            return ranks[1:1+top_k]

        else:
            # perform PMI retrieval
            ranks = self.PMI_sampling(int(symp_idx))
            return ranks[:top_k]


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

    def PMI_sampling(self, symp_idx):
        """Sampling through the PMI method, that is based on the co-occurence of symptoms.
        """
        all_scores = self.symp2symp[symp_idx]

        score_vals = all_scores.data
        score_inds = all_scores.indices
        sort_idx = np.argsort(-score_vals)
        return score_inds[sort_idx]



    def build_pmi_matrix(self, prefix):
        symp2symp = defaultdict(list)
        symp2count = defaultdict(int)

        pmi_datapath = os.path.join(prefix, "train/data.txt")
        fin = open(pmi_datapath, "r", encoding="utf-8")
        for line in fin.readlines():
            line_data = line.strip().split("\t")
            symps = line_data[2:]
            for symp in symps:
                symp2symp[symp].extend(symps)
                symp2count[symp] += 1

        self.sympcount = symp2count
        csr_data, csr_col, csr_row = [],[],[]
        for symp in range(1,self.num_symp):
            coc_count = pd.value_counts(symp2symp[str(symp)])
            csr_data.extend(coc_count.values.tolist())
            csr_row.extend([symp]*len(coc_count))
            csr_col.extend(coc_count.index.values.astype(int).tolist())


        self.symp2symp = sparse.csr_matrix((csr_data,(csr_row,csr_col)))
        sparse.save_npz(self.pmi_mat_path, self.symp2symp)
        np.save(self.symp_count_path ,self.symp2count)
        print("Done PMI Mat Building.")

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


if __name__ == '__main__':
    import fire
    fire.Fire()




