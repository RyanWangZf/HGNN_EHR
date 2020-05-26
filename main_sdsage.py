# -*- coding: utf-8 -*-
"""python -u main_sdsage.py train --use_gpu=False
"""

from collections import defaultdict
import pdb
import time
import os

from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

import scipy.sparse as sp

from dataset import ehr
from utils import setup_seed, collate_fn
from config import default_config
from utils import now, parse_kwargs, parse_data_model, EarlyStopping
from model import DSD_sampler

class HGNN_SDS(nn.Module):
    def __init__(self, *args, **kwargs):
        super(HGNN_SDS, self).__init__()
        self.num_symp = kwargs["num_symp"]
        self.num_dise = kwargs["num_dise"]
        self.use_gpu = kwargs["use_gpu"]

        embed_dim = kwargs["symp_embedding_dim"]
        # init embeddings
        self.symp_embeds = torch.nn.Embedding(
            self.num_symp+1,
            embed_dim,
            padding_idx=0,)
        self.dise_embeds = torch.nn.Embedding(
            self.num_dise+1,
            embed_dim,
            padding_idx=0,)

        self.w2 = nn.Linear(embed_dim, embed_dim)
        self.w1 = nn.Linear(2*embed_dim, embed_dim)
        self.w0 = nn.Linear(2*embed_dim, embed_dim)
        self.w_last = nn.Linear(embed_dim, embed_dim)

        self.act_fn = nn.ReLU()
        self.num_sds_1_hop = 1
        self.num_sds_2_hop = 20

    def load_symp_embed(self, w2v_path="./ckpt/w2v"):
        """Load pretrained symptom embeddings from Word2Vec model.
        """
        from gensim.models import Word2Vec
        embed_dim = self.embed_dim

        w2v_model = Word2Vec.load(w2v_path)
        # init embedding matrix
        w2v_list = [[0]*embed_dim]
        for i in range(1,self.num_symp+1):
            w2v_list.append(w2v_model.wv[str(i)])

        w2v_param = torch.FloatTensor(w2v_list)
        self.symp_embeds.weight.data.copy_(w2v_param)
        # freeze the symptom embeddings
        self.symp_embeds.requires_grad = False
        print("Load pretrained symptom embeddings from", w2v_path)

    def forward(self, x, data):
        target_emb_symp = self.symp_embeds(x)

        z_dise_1_list = []
        for i in range(self.num_sds_1_hop):
            sds_2_hop_nb = torch.LongTensor(data["sds_2_{}".format(int(i))])
            if self.use_gpu:
                sds_2_hop_nb = sds_2_hop_nb.cuda()

            emb_symp_2_hop = self.symp_embeds(sds_2_hop_nb) # ?, num_2_hop, k
            z_symp_2 = self.w2(emb_symp_2_hop) # ?, 100, k
            z_symp_2 = self.act_fn(z_symp_2).mean(1) # ?, 100
            z_symp_2 = F.normalize(z_symp_2, p=1, dim=1)

            # concat with one hop neighbor
            sds_1_hop_nb = torch.LongTensor(data["sds_1"][:,i])
            if self.use_gpu:
                sds_1_hop_nb = sds_1_hop_nb.cuda()
            
            emb_dise = self.dise_embeds(sds_1_hop_nb) # ?, k

            z_cat = torch.cat([emb_dise, z_symp_2], axis=1)
            z_dise_1 = self.act_fn(self.w1(z_cat))
            z_dise_1 = F.normalize(z_dise_1, p=2, dim=1)
            z_dise_1_list.append(torch.unsqueeze(z_dise_1,1))

        z_dise_1_all = torch.cat(z_dise_1_list, 1)
        z_dise_1_all = z_dise_1_all.mean(1)

        z_symp_1 = torch.cat([target_emb_symp, z_dise_1_all], axis=1)
        z_symp_1 = F.normalize(self.act_fn(self.w0(z_symp_1)),p=2,dim=1)

        # final dense layer
        z_symp_last = self.w_last(z_symp_1)

        return z_symp_last

    def fill_zero(self, ar, target_num):
        ar_len = len(ar)
        assert target_num >= ar_len
        if ar_len < target_num:
            ar = np.r_[ar, ["0"]*(target_num - ar_len)]
        return ar

    def random_select(self, ar, target_num):
        all_idx = np.arange(len(ar))
        np.random.shuffle(all_idx)
        return ar[all_idx[:target_num]]


class SDS_sampler:
    """Given targeted array of symptoms, sampling their neighborhood based S-D-S path.
    """
    def __init__(self, prefix, use_pmi=True):
        self.prefix = prefix
        # load maps
        if use_pmi:
            dise2symp = np.load(os.path.join(prefix,"dise2symp_s_n.npy"),allow_pickle=True).item()
            self.dise2symp = {}
            for k in dise2symp.keys():
                self.dise2symp[str(k)] = dise2symp[k]
                
        else:
            self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy"),allow_pickle=True).item()

        self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy"),allow_pickle=True).item()
        self.num_dise = len(self.dise2symp.keys())
        self.num_symp = len(self.symp2dise.keys())

    def __call__(self, label, num_1_hop, num_2_hop):
        return self.sampling(label, num_1_hop, num_2_hop)

    def sampling(self, label, num_1_hop, num_2_hop):
        """Neighbor sampling for SDS metapath.
        """
        symp_ar = label
        data_dict = defaultdict(list)

        for i in range(len(symp_ar)):
            symp = symp_ar[i]
            sds_1_hop_nb = self.symp2dise[str(symp)]
            sds_1_hop_nb = self.random_select(sds_1_hop_nb, num_1_hop)
            sds_1_hop_nb = self.fill_zero(sds_1_hop_nb, num_1_hop)

            data_dict["sds_1"].append(sds_1_hop_nb)
            dsd_2_hop_nb_list = []

            for i in range(len(sds_1_hop_nb)):
                dise = sds_1_hop_nb[i]
                sds_2_hop_nb = self.dise2symp.get(dise)
                sds_2_hop_nb = self.random_select(sds_2_hop_nb, num_2_hop)
                sds_2_hop_nb = self.fill_zero(sds_2_hop_nb, num_2_hop)
                data_dict["sds_2_{}".format(i)].append(sds_2_hop_nb)

        for k in data_dict.keys():
            data_dict[k] = np.array(data_dict[k]).astype(int)

        return data_dict

    def fill_zero(self, ar, target_num):
        ar_len = len(ar)
        assert target_num >= ar_len
        if ar_len < target_num:
            ar = np.r_[ar, ["0"]*(target_num - ar_len)]
        return ar

    def random_select(self, ar, target_num):
        all_idx = np.arange(len(ar))
        np.random.shuffle(all_idx)
        res = ar[all_idx[:target_num]]
        return res


def train(**kwargs):
    setup_seed(2020)
    model_param = default_config()
    model_param = parse_kwargs(model_param, kwargs)

    # load training data
    train_data = ehr.EHR("dataset/EHR","train")
    train_data_loader = DataLoader(train_data, 
        model_param["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_fn)

    # init model
    data_model_param = parse_data_model(train_data)
    model_param.update(data_model_param)
    use_gpu = model_param["use_gpu"]

    gnn = HGNN_SDS(**model_param)
    if model_param["w2v"] is not None:
        # load w2v data
        gnn.load_symp_embed(model_param["w2v"])

    if use_gpu:
        gnn.cuda()

    print("Model Inited.")

    sds_sampler = SDS_sampler("dataset/EHR")

    # load pmi ss mat
    symp2symp_mat = sp.load_npz(os.path.join("dataset/EHR","pmi_ss_mat.npz"))
    symp2symp_mat.setdiag(0)

    # total number of symptoms
    num_total_batch = gnn.num_symp // model_param["batch_size"]
    all_symp_index = np.arange(1,gnn.num_symp+1)

    lambda_hard_r = lambda epoch: epoch * model_param["hard_ratio"] / model_param["num_epoch"]

    # build hard map and pos map
    symp2symp_hard_map = [0]
    symp2symp_pos_map = [0]
    for k in all_symp_index:
        symp2symp_b_ar = symp2symp_mat[k].toarray().flatten()
        max_index = np.argmax(symp2symp_b_ar)
        if max_index == 0:
            symp2symp_pos_map.append(np.random.randint(1, k))
            symp2symp_hard_map.append(np.random.randint(1, k))

        else:
            symp2symp_pos_map.append(max_index)
            symp2symp_b_ar[max_index] = -1
            max_2nd_index = np.argmax(symp2symp_b_ar)
            if max_2nd_index == 0:
                symp2symp_hard_map.append(np.random.randint(1, k))
            else:
                symp2symp_hard_map.append(max_2nd_index)


    symp2symp_hard_map = np.array(symp2symp_hard_map)
    symp2symp_pos_map = np.array(symp2symp_pos_map)
    print("Pos / Hard symptom map Inited.")

    optimizer = torch.optim.Adam(gnn.parameters(),lr=model_param["lr"], weight_decay=model_param["lr"])

    for epoch in range(model_param["num_epoch"]):
        total_loss = 0
        last_total_loss = -1e10
        gnn.train()
        np.random.shuffle(all_symp_index)

        hard_ratio = lambda_hard_r(epoch)

        for idx in range(num_total_batch):
            batch_symp = all_symp_index[idx*model_param["batch_size"]:(idx+1)*model_param["batch_size"]]

            # get pos symp and neg symp
            pos_symp = symp2symp_pos_map[batch_symp]

            # sample neg
            neg_symp = np.random.randint(1,gnn.num_symp,model_param["batch_size"])

            # cope with overlapping in pos and neg symps
            overlap_index = (neg_symp == pos_symp)
            overlap_symp = neg_symp[overlap_index]
            neg_symp[overlap_index] = symp2symp_hard_map[overlap_symp]

            if hard_ratio > 0:
                num_hard = int(hard_ratio * model_param["batch_size"])
                neg_symp[:num_hard] = symp2symp_hard_map[neg_symp[:num_hard]]

            batch_symp_ts = torch.LongTensor(batch_symp)
            pos_symp_ts = torch.LongTensor(pos_symp)
            neg_symp_ts = torch.LongTensor(neg_symp)

            if model_param["use_gpu"]:
                batch_symp_ts = batch_symp_ts.cuda()
                pos_symp_ts = pos_symp_ts.cuda()
                neg_symp_ts = neg_symp_ts.cuda()

            # forward batch symp
            batch_symp_data = sds_sampler(batch_symp,1,20)
            symp_emb = gnn.forward(batch_symp_ts, batch_symp_data)

            pos_symp_data = sds_sampler(pos_symp,1,20)
            pos_emb = gnn.forward(pos_symp_ts, pos_symp_data)

            neg_symp_data = sds_sampler(neg_symp,1,20)
            neg_emb = gnn.forward(neg_symp_ts, neg_symp_data)

            # create loss
            scores = symp_emb.mul(pos_emb).sum(1) - symp_emb.mul(neg_emb).sum(1) + 1.0
            scores[scores < 0] = 0

            loss = scores.mean()

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print("{} Epoch {}/{}: train loss: {:.6f}".format(now(),epoch+1,
            model_param["num_epoch"],total_loss))

        if total_loss - last_total_loss > 0:
            print("Loss stops to decrease, converge.")

        last_total_loss = total_loss

    # save model
    torch.save(gnn.state_dict(), "./ckpt/sds_gnn.pt")
    print("Model saved.")


if __name__ == '__main__':
    import fire
    fire.Fire()