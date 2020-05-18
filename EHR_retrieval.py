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

        assert mode in ["sus", "sds", "mix", "pmi", "gpmi"]
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

        if mode in ["pmi","gpmi"]:
            # init a PMI matrix that has shape M X M
            # we'd better make it a sparse matrix
            # if we pick the graphPMI (gpmi) method, we need an additional S-D PMI matrix.

            # read data
            self.pmi_ss_path = os.path.join(prefix, "pmi_ss_mat.npz")
            self.pmi_sd_path = os.path.join(prefix, "pmi_sd_mat.npz")
            self.symp_count_path = os.path.join(prefix, "sympcount.npy")
            self.dise_count_path = os.path.join(prefix, "disecount.npy")
            self.dise2symp_path = os.path.join(prefix, "dise2symp.npy")
            if os.path.exists(self.pmi_ss_path):
                print("Load PMI Mat from", self.pmi_ss_path)
                self.symp2symp = sparse.load_npz(self.pmi_ss_path)
                self.symp2dise = sparse.load_npz(self.pmi_sd_path)
                self.sympcount = np.load(self.symp_count_path, allow_pickle=True).item()
                self.disecount = np.load(self.dise_count_path, allow_pickle=True).item()
                self.symp2symp.setdiag(0)
            else:
                print("Build PMI Mat.")
                self.build_pmi_matrix(prefix)
                self.symp2symp.setdiag(0)

            # build symp count array
            c_ar, i_ar = [], []
            for k,v in self.sympcount.items():
                c_ar.append(v)
                i_ar.append(int(k))
            sympcount_mat = sparse.csr_matrix((c_ar, (i_ar, [0]*len(i_ar))))
            self.sympcount_ar = sympcount_mat.toarray().flatten()
            self.num_all_symp = self.sympcount_ar.sum()

            # build dise count array
            c_ar, i_ar = [], []
            for k,v in self.disecount.items():
                c_ar.append(v)
                i_ar.append(int(k))
            disecount_mat = sparse.csr_matrix((c_ar, (i_ar, [0]*len(i_ar))))
            self.disecount_ar = disecount_mat.toarray().flatten()
            self.num_all_dise = self.disecount_ar.sum()  

            self.dise2symp = np.load(self.dise2symp_path, allow_pickle=True).item()


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

        if self.mode == "pmi":
            # perform PMI retrieval
            ranks = self.PMI_sampling(int(symp_idx))
            return ranks[:top_k]

        if self.mode == "gpmi":
            # perform PMI retrieval
            ranks = self.GPMI_sampling(int(symp_idx), top_k)
            return ranks


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

        num_A = self.sympcount_ar[symp_idx]
        num_B = self.sympcount_ar[score_inds]

        pmi_ind = np.log2((self.num_all_symp * score_vals + 1e-8) / (num_A * num_B + 1e-8))
        Z = - np.log2((score_vals +1e-8)/ (self.num_all_symp))
        npmi_ind = pmi_ind / Z # normalized pointwise mutual information

        sort_idx = np.argsort(-pmi_ind)

        return score_inds[sort_idx]

    def GPMI_sampling(self, symp_idx, top_k):
        """Sampling through the graphical PMI method, based on S - D - S meta-path.
        """
        # find S - D scores
        sd_scores = self.symp2dise[symp_idx]
        sd_score_vals = sd_scores.data
        sd_score_inds = sd_scores.indices

        if len(sd_score_vals) <= 1:
            # only one disease is connected to this symptom
            # directly retrieve symptoms under this disease
            sub_symps = self.dise2symp[str(sd_score_inds[0])].astype(int)
            sub_symps = np.setdiff1d(sub_symps, [symp_idx]) ## delete target itself
            num_A = self.sympcount_ar[symp_idx]
            num_B = self.sympcount_ar[sub_symps]
            co_oc_count = self.symp2symp[symp_idx].toarray()[0][sub_symps]
            pmi =  np.log2((self.num_all_symp * co_oc_count + 1e-8) / (num_A * num_B + 1e-8))
            Z = - np.log2((co_oc_count+1e-8) / (self.num_all_symp+1e-8))
            npmi = pmi / Z
            sort_idx = np.argsort(-npmi)
            sort_sub_symps = sub_symps[sort_idx]
            return sort_sub_symps[:top_k]

        else:  
            p_dise = self.disecount_ar[sd_score_inds] / self.num_all_dise
            p_symp = self.sympcount_ar[symp_idx] / self.num_all_symp
            pmi_sd = np.log2((1e-8 + sd_score_vals/self.num_all_dise)/(1e-8 + p_dise * p_symp))
            Z = - np.log2((sd_score_vals +1e-8)/(1e-8+ self.num_all_dise))
            npmi_sd = pmi_sd / Z
            # select symps according to npmi sd
            target_num = np.ceil(top_k * softmax(npmi_sd)).astype(int)

            alternative_symp_list = []
            for j, dise in enumerate(sd_score_inds):
                sub_symps = self.dise2symp[str(dise)].astype(int)
                sub_symps = np.setdiff1d(sub_symps, [symp_idx]) # delete target itself

                if j > 0:
                    # remove already picked symps
                    sub_symps = np.setdiff1d(sub_symps, alternative_symp_list)

                num_A = self.sympcount_ar[symp_idx]
                num_B = self.sympcount_ar[sub_symps]
                co_oc_count = self.symp2symp[symp_idx].toarray()[0][sub_symps]
                pmi =  np.log2((self.num_all_symp * co_oc_count + 1e-8) / (num_A * num_B + 1e-8))
                Z = - np.log2((co_oc_count+1e-8) / (self.num_all_symp+1e-8))
                npmi = pmi / Z
                sort_idx = np.argsort(-npmi)
                sort_sub_symps = sub_symps[sort_idx]
                alternative_symp_list.extend(sort_sub_symps[:target_num[j]].tolist())

            return alternative_symp_list[:top_k]

    def build_pmi_matrix(self, prefix):
        symp2symp = defaultdict(list)
        symp2dise = defaultdict(list)
        symp2count = defaultdict(int)
        dise2count = defaultdict(int)

        pmi_datapath = os.path.join(prefix, "train/data.txt")
        fin = open(pmi_datapath, "r", encoding="utf-8")
        for line in fin.readlines():
            line_data = line.strip().split("\t")
            diseid = line_data[1]
            symps = line_data[2:]
            dise2count[diseid] += 1
            for symp in symps:
                symp2symp[symp].extend(symps)
                symp2count[symp] += 1
                symp2dise[symp].append(diseid)

        self.sympcount = symp2count
        csr_data, csr_col, csr_row = [],[],[]
        csr_sd_data, csr_sd_col, csr_sd_row = [],[],[]
        for symp in range(1,self.num_symp):
            coc_count = pd.value_counts(symp2symp[str(symp)])
            csr_data.extend(coc_count.values.tolist())
            csr_row.extend([symp]*len(coc_count))
            csr_col.extend(coc_count.index.values.astype(int).tolist())

            coc_sd_count = pd.value_counts(symp2dise[str(symp)])
            csr_sd_data.extend(coc_sd_count.values.tolist())
            csr_sd_row.extend([symp]*len(coc_sd_count))
            csr_sd_col.extend(coc_sd_count.index.values.astype(int).tolist())

        self.symp2symp = sparse.csr_matrix((csr_data,(csr_row,csr_col)))
        self.symp2dise = sparse.csr_matrix((csr_sd_data, (csr_sd_row, csr_sd_col)))
        self.disecount = dise2count
        sparse.save_npz(self.pmi_ss_path, self.symp2symp)
        sparse.save_npz(self.pmi_sd_path, self.symp2dise)
        np.save(self.symp_count_path ,self.sympcount)
        np.save(self.dise_count_path, self.disecount)
        print("Done PMI Mat Building.")


def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


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




