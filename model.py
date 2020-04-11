# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import defaultdict

import pdb
import os

class HGNN(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(HGNN, self).__init__()
        self.num_symp = kwargs["num_symp"]
        self.num_dise = kwargs["num_dise"]

        self._parse_metapath_param(kwargs)
        self.use_gpu = kwargs["use_gpu"]

        layer_size_dsd = kwargs["layer_size_dsd"]
        layer_size_usu = kwargs["layer_size_usu"]

        # now we make embedding dim same !!
        assert kwargs["symp_embedding_dim"] == kwargs["dise_embedding_dim"]
        embed_dim = kwargs["symp_embedding_dim"]
        for l in layer_size_usu:
            assert l == embed_dim
        for l in layer_size_dsd:
            assert l == embed_dim

        # build hard map for negative sampling
        self.hard_neg_map = {}
        for i in range(self.num_dise+1):
            self.hard_neg_map[i] = i+1
        self.hard_neg_map[i] = i-1

        # init embeddings
        self.symp_embeds = torch.nn.Embedding(
            self.num_symp+1,
            embed_dim,
            padding_idx=0,
            )

        self.dise_embeds = torch.nn.Embedding(
            self.num_dise+1,
            embed_dim,
            padding_idx=0)

        # init weights based on meta-path params

        # for D-S-D
        self.linear_dsd_2_1 = torch.nn.Linear(
            embed_dim, 
            layer_size_dsd[0], 
            bias=False,
            )

        self.linear_dsd_2_2 = torch.nn.Linear(
            embed_dim,
            layer_size_dsd[0],
            bias=False,)

        self.linear_dsd_1_1 = torch.nn.Linear(
            layer_size_dsd[0], 
            layer_size_dsd[1],
            bias=False,)

        self.linear_dsd_1_2 = torch.nn.Linear(
            layer_size_dsd[0],
            layer_size_dsd[1],
            bias=False,)

        # for U-S-U
        self.linear_usu_3 = torch.nn.Linear(
            embed_dim, 
            layer_size_usu[0],
            bias=False,)

        self.linear_usu_2_1 = torch.nn.Linear(
            embed_dim, 
            layer_size_usu[0],
            bias=False,)

        self.linear_usu_2_2 = torch.nn.Linear(
            embed_dim, 
            layer_size_usu[0],
            bias=False,)

        self.linear_usu_1 = torch.nn.Linear(
            embed_dim, 
            layer_size_usu[0],
            bias=False,)

        self.act_fn = torch.nn.LeakyReLU(0.2)
        self.dropout_fn = torch.nn.Dropout(p=kwargs["dropout_ratio"])

    def forward(self, feat, label, dsd_sampler=None):
        """Inputs:
        symp: an array of dict, each dict is a sample.
        dise: an array of int.
        """

        # involve all data in batch dict
        data = self._array2dict(feat)
        ts_label = torch.LongTensor(label)
        
        if self.use_gpu:
            ts_label = ts_label.cuda()

        # for path dise - symp - dise
        emb_dise = self.forward_dsd(data, ts_label) # ?, k

        # for path user - symp - user - symp
        emb_user = self.forward_usu(data) # ?, k

        # pred score
        pred_score = emb_dise.mul(emb_user).sum(1)

        # if self.training = True, we need negative sampling for paired BPR loss
        if self.training:
            label_neg = self._neg_sampling_dise(label)
            
            if self.use_gpu:
                ts_label_neg = torch.LongTensor(label_neg).cuda()
            else:
                ts_label_neg = torch.LongTensor(label_neg)
                
            neg_data = dsd_sampler.sampling(label_neg, 
                self.num_dsd_1_hop,
                self.num_dsd_2_hop)

            for k in neg_data.keys():
                if self.use_gpu:
                    neg_data[k] = torch.LongTensor(neg_data[k]).cuda()
                else:
                    neg_data[k] = torch.LongTensor(neg_data[k])

            neg_emb_dise = self.forward_dsd(neg_data, ts_label_neg)
            neg_pred_score = neg_emb_dise.mul(emb_user).sum(1)

            return pred_score, neg_pred_score, emb_user, emb_dise, neg_emb_dise
        
        else:
            return pred_score

    def forward_dsd(self, data, label):

        target_emb_d = self.dise_embeds(label)

        # for d-s-d
        emb_s = self.symp_embeds(data["dsd_1"]) # ?, 10, 16

        emb_s_1_list = []

        for i in range(self.num_dsd_1_hop):
            dise_2_hop = data["dsd_2_{}".format(i)]

            # m(s-d)
            emb_d = self.dise_embeds(dise_2_hop) # ?, 2, k
            emb_s_ = emb_s[:,i]

            m_s_d = self.linear_dsd_2_1(emb_d) + \
                self.linear_dsd_2_2(emb_d.mul(torch.unsqueeze(emb_s_, 1))) # ?, 2, k

            m_s_s = self.linear_dsd_2_1(emb_s_) # ?,k
            m_s_s = self.dropout_fn(m_s_s)

            m_s_d = self._avg_on_real_neighbor(m_s_d, dise_2_hop) # ?,k
            m_s_d = self.dropout_fn(m_s_d)

            emb_s_1_ = self.act_fn(m_s_s + m_s_d)
            emb_s_1_list.append(torch.unsqueeze(emb_s_1_,1))

        emb_s_1 = torch.cat(emb_s_1_list, 1) # ?, 10, k
        
        m_d_s = self.linear_dsd_1_1(emb_s_1) + \
            self.linear_dsd_1_2(emb_s_1.mul(torch.unsqueeze(target_emb_d,1))) # ?, 10, k
        
        symp_1_hop = data["dsd_1"]

        m_d_s = self._avg_on_real_neighbor(m_d_s, symp_1_hop) # ?, k
        m_d_d = self.linear_dsd_1_1(target_emb_d) # ?, k

        emb_d_last = self.act_fn(m_d_s + m_d_d)

        return emb_d_last

    def forward_usu(self, data):

        emb_s_1_list = []

        emb_s_1 = self.symp_embeds(data["usu_1"]) #?, 5, k
        for i in range(self.num_usu_1_hop):
            emb_u_2_list = []
            emb_s_1_ = emb_s_1[:,i] # ?, k

            for j in range(self.num_usu_2_hop):
                name = "usu_3_{}".format(i*self.num_usu_1_hop+j)
                symp_3_hop = data[name]
                emb_s_3 = self.symp_embeds(symp_3_hop) # ?,5, k
                m_u_s = self.linear_usu_3(emb_s_3) # ?, 5, k
                m_u_s = self._avg_on_real_neighbor(m_u_s, symp_3_hop)
                m_u_s = self.dropout_fn(m_u_s)

                # no m_u_u here because we dont have users embeddings
                emb_u_2_ = self.act_fn(m_u_s) # ?, k

                emb_u_2_list.append(torch.unsqueeze(emb_u_2_, 1))

            emb_u_2 = torch.cat(emb_u_2_list, 1) # ?, 5, k
            user_2_hop = data["usu_2_{}".format(i)]

            m_s_u = self.linear_usu_2_1(emb_u_2) + \
                self.linear_usu_2_2(emb_u_2.mul(torch.unsqueeze(emb_s_1_,1)))

            m_s_u = self._avg_on_real_neighbor(m_s_u, user_2_hop) # ?, k
            m_s_u = self.dropout_fn(m_s_u)

            m_s_s = self.linear_usu_2_1(emb_s_1_) # ?, k
            m_s_s = self.dropout_fn(m_s_s)

            emb_s_1_list.append(torch.unsqueeze(
                    self.act_fn(m_s_s + m_s_u),1))

        emb_s_1_last = torch.cat(emb_s_1_list, 1) # ?, 5, k
        symp_1_hop = data["usu_1"]
        m_u_s = self.linear_usu_1(emb_s_1_last)
        m_u_s = self._avg_on_real_neighbor(m_u_s, symp_1_hop) # ?, k

        # no m_u_u because we dont have users' embeddings
        emb_u_last = self.act_fn(m_u_s)
        return emb_u_last

    def gen_all_dise_emb(self, dsd_sampler):
        """Generate alternative embeddings for all 27 diseases, 
        namely emb_dise.
        """
        self.eval()
        dise_label = np.arange(1, self.num_dise+1)
        dise_data = dsd_sampler.sampling(dise_label, 
            self.num_dsd_1_hop,
            self.num_dsd_2_hop)

        for k in dise_data.keys():
            if self.use_gpu:
                dise_data[k] = torch.LongTensor(dise_data[k]).cuda()
            else:
                dise_data[k] = torch.LongTensor(dise_data[k])
                
        if self.use_gpu:
            dise_label = torch.LongTensor(dise_label).cuda()
        else:
            dise_label = torch.LongTensor(dise_label)

        with torch.no_grad():
            emb_dise = self.forward_dsd(dise_data, dise_label)

        return emb_dise

    def rank_query(self, user_query, emb_dise, usu_sampler, top_k=3):
        self.eval()

        user_data = usu_sampler(user_query, self.num_usu_1_hop,
            self.num_usu_2_hop, self.num_usu_3_hop)

        for k in user_data.keys():
            if self.use_gpu:
                user_data[k] = torch.LongTensor(user_data[k]).cuda()
            else:
                user_data[k] = torch.LongTensor(user_data[k])

        with torch.no_grad():
            emb_user = self.forward_usu(user_data)

        pred_score = emb_user.matmul(emb_dise.T)

        # ranking
        pred_rank_top_k = torch.argsort(-pred_score,1)[:,:top_k] + 1

        return pred_rank_top_k.detach().cpu().numpy()

    def _array2dict(self, feat):
        """Transform a batch of tuples to a
        dict contains batch of longtensor.
        """
        key_list = list(feat[0].keys())
        data = defaultdict(list)

        for x in feat:
            for k in key_list:
                data[k].append(x[k])

        for k in key_list:
            if self.use_gpu:
                data[k] = torch.LongTensor(data[k]).cuda()
            else:
                data[k] = torch.LongTensor(data[k])


        return data

    def _avg_on_real_neighbor(self, msg, ts):
        real_num_2_hop = torch.unsqueeze((ts != 0).sum(1).float(),1)
        weight = 1 / (real_num_2_hop + 1e-8)
        weight[weight == 1e8] = 0
        
        return msg.sum(1).mul(weight)

    def _neg_sampling_dise(self, label):
        """Inputs:
        label: array of diseases
        
        Outputs:
        neg_dise: array of negative sampling diseases

        """
        num_sample = len(label)
        neg_dise = np.random.randint(1, self.num_dise+1, num_sample)
        same_idx = (label == neg_dise)

        if np.sum(same_idx) > 0:
            rep_neg_dise = [self.hard_neg_map[x] for x in label[same_idx]]
            neg_dise[same_idx] = rep_neg_dise

        return neg_dise

    def _parse_metapath_param(self, param):
        # for DSD
        self.num_dsd_1_hop = param["num_dsd_1_hop"]
        self.num_dsd_2_hop = param["num_dsd_2_hop"] 

        # for USU
        self.num_usu_1_hop = param["num_usu_1_hop"] 
        self.num_usu_2_hop = param["num_usu_2_hop"]  
        self.num_usu_3_hop = param["num_usu_3_hop"]  
        return


class USU_sampler:
    def __init__(self, prefix):
        self.prefix = prefix
        # load maps
        self.symp2user = np.load(os.path.join(prefix,"symp2user.npy"),allow_pickle=True).item()
        self.user2symp = np.load(os.path.join(prefix,"user2symp.npy"),allow_pickle=True).item()

    def __call__(self, symps, num_USU_1_hop, num_USU_2_hop, num_USU_3_hop):
        return self.sampling(symps, num_USU_1_hop, num_USU_2_hop, num_USU_3_hop)

    def sampling(self, symps, num_USU_1_hop, num_USU_2_hop, num_USU_3_hop):
        """Neighbor sampling for U-S-U metapath.
        Inputs an array of symptoms, outputs the USU neighbors
        for generating user embeddings.
        """

        data_dict = defaultdict(list)

        for s in range(len(symps)):
            symp_list = symps[s]

            # ---------------
            # for USU path
            usu_1_hop_nb = symp_list.astype(str)
            usu_1_hop_nb = self.random_select(usu_1_hop_nb, num_USU_1_hop)
            usu_1_hop_nb = self.fill_zero(usu_1_hop_nb, num_USU_1_hop)
            data_dict["usu_1"].append(usu_1_hop_nb)

            # user's symptoms is usu_1 !
            data_dict["symp"].append(usu_1_hop_nb)

            for i in range(len(usu_1_hop_nb)):
                symp = usu_1_hop_nb[i]
                usu_2_hop_nb = self.symp2user.get(symp)
                if usu_2_hop_nb is None:
                    usu_2_hop_nb = []
                else:
                    usu_2_hop_nb = self.random_select(usu_2_hop_nb, num_USU_2_hop)
                
                usu_2_hop_nb = self.fill_zero(usu_2_hop_nb, num_USU_2_hop)
                data_dict["usu_2_{}".format(i)].append(usu_2_hop_nb)

                for j in range(len(usu_2_hop_nb)):
                    uid = usu_2_hop_nb[j]
                    usu_3_hop_nb = self.user2symp.get(uid)
                    if usu_3_hop_nb is None:
                        usu_3_hop_nb = []
                    else:
                        usu_3_hop_nb = self.random_select(usu_3_hop_nb, num_USU_3_hop)

                    usu_3_hop_nb = self.fill_zero(usu_3_hop_nb, num_USU_3_hop)
                    data_dict["usu_3_{}".format(i*num_USU_2_hop+j)].append(usu_3_hop_nb)

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
        return ar[all_idx[:target_num]]



class DSD_sampler:
    """Given targeted array of diseases,
    sampling their neighborhood based D-S-D path.
    """
    def __init__(self, prefix):
        self.prefix = prefix
        # load maps
        self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy"),allow_pickle=True).item()
        self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy"),allow_pickle=True).item()
        self.num_dise = len(self.dise2symp.keys())
        self.num_symp = len(self.symp2dise.keys())

    def __call__(self, label, num_DSD_1_hop, num_DSD_2_hop):
        return self.sampling(label, num_DSD_1_hop, num_DSD_2_hop)

    def sampling(self, label, num_DSD_1_hop, num_DSD_2_hop):
        """Neighbor sampling for D-S-D metapath.
        """
        dise_ar = label

        data_dict = defaultdict(list)

        for l in range(len(dise_ar)):
            # ---------------
            # for DSD path
            dise = dise_ar[l]
            dsd_1_hop_nb = self.dise2symp[str(dise)]
            dsd_1_hop_nb = self.random_select(dsd_1_hop_nb, num_DSD_1_hop)
            dsd_1_hop_nb = self.fill_zero(dsd_1_hop_nb, num_DSD_1_hop)

            data_dict["dsd_1"].append(dsd_1_hop_nb)

            dsd_2_hop_nb_list = []

            for i in range(len(dsd_1_hop_nb)):
                symp = dsd_1_hop_nb[i]
                dsd_2_hop_nb = self.symp2dise.get(symp)
                if dsd_2_hop_nb is None:
                    dsd_2_hop_nb = []
                else:
                    dsd_2_hop_nb = self.random_select(dsd_2_hop_nb, num_DSD_2_hop)
                
                dsd_2_hop_nb = self.fill_zero(dsd_2_hop_nb, num_DSD_2_hop)

                data_dict["dsd_2_{}".format(i)].append(dsd_2_hop_nb)
            # ---------------
        
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
        return ar[all_idx[:target_num]]