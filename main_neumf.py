# -*- coding: utf-8 -*-
import torch
import numpy as np
from collections import defaultdict

import pdb
from torch.utils.data import DataLoader
import torch.nn.functional as F

from dataset import ehr
from utils import now, parse_kwargs, parse_data_model, EarlyStopping
from utils import setup_seed, collate_fn
from config import default_config

class NeuMF(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(NeuMF, self).__init__()
        self.num_symp = kwargs["num_symp"]
        self.num_dise = kwargs["num_dise"]
        self.use_gpu = kwargs["use_gpu"]
        embed_dim = kwargs["symp_embedding_dim"]

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

        self.linear_1 = torch.nn.Linear(embed_dim*2, embed_dim)
        self.linear_2 = torch.nn.Linear(embed_dim, 1)


    def forward(self, feat, label):

        data = self._array2dict(feat)
        ts_label = torch.LongTensor(label)

        if self.use_gpu:
            ts_label = ts_label.cuda()

        # get emb dise
        emb_dise = self.dise_embeds(ts_label)

        # get emb symps
        symps = data["symp"]
        real_num_neighbor = torch.unsqueeze((symps != 0).sum(1).float(),1)
        emb_symp = self.symp_embeds(data["symp"])

        weight = 1 / (real_num_neighbor+1e-8)
        weight[weight >= 1e8] = 0
        emb_user = emb_symp.sum(1).mul(weight)

        emb_u_s = torch.cat([emb_user, emb_dise], axis=1)
        h_u_s = F.relu(emb_u_s)
        h_u_s = self.linear_1(h_u_s)
        h_u_s = F.relu(h_u_s)
        pred_score = self.linear_2(h_u_s) # ?, 1


        # pred_score = emb_dise.mul(emb_user).sum(1)

        if self.training:
            label_neg = self._neg_sampling_dise(label)

            if self.use_gpu:
                ts_label_neg = torch.LongTensor(label_neg).cuda()
            else:
                ts_label_neg = torch.LongTensor(label_neg)
            neg_emb_dise = self.dise_embeds(ts_label_neg)
            
            neg_emb_u_s = torch.cat([emb_user, neg_emb_dise], axis=1)
            neg_h_u_s = F.relu(neg_emb_u_s)
            neg_h_u_s = self.linear_1(neg_h_u_s)
            neg_h_u_s = F.relu(neg_h_u_s)
            neg_pred_score = self.linear_2(neg_h_u_s) # ?, 1

            # neg_pred_score = neg_emb_dise.mul(emb_user).sum(1)
            return pred_score, neg_pred_score, emb_user, emb_dise, neg_emb_dise

        else:
            return pred_score

    def forward_user(self, feat):
        data = self._array2dict(feat)
        symps = data["symp"]
        real_num_neighbor = torch.unsqueeze((symps != 0).sum(1).float(),1)
        emb_symp = self.symp_embeds(data["symp"])
        weight = 1 / (real_num_neighbor+1e-8)
        weight[weight >= 1e8] = 0
        emb_user = emb_symp.sum(1).mul(weight)
        return emb_user


    def forward_user_dise(self, emb_user, emb_dise):
        h_result = []
        for i in range(emb_dise.shape[0]):
            emb_u_s = torch.cat([emb_user, emb_dise[i].repeat(emb_user.shape[0],1)], axis=1)
            h = F.relu(emb_u_s)
            h = self.linear_1(h)
            h = F.relu(h)
            h = self.linear_2(h)
            h_result.append(h)

        pred = torch.cat(h_result, 1)
        return pred

    def gen_all_dise_emb(self):
        self.eval()
        dise_label = np.arange(1, self.num_dise+1)

        if self.use_gpu:
            dise_label = torch.LongTensor(dise_label).cuda()
        else:
            dise_label = torch.LongTensor(dise_label)


        emb_dise = self.dise_embeds(dise_label)

        return emb_dise

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

def train(**kwargs):
    setup_seed(2020)

    model_param = default_config()
    model_param = parse_kwargs(model_param, kwargs)

    # load training data
    train_data = ehr.EHR("dataset/EHR","train")
    train_data_loader = DataLoader(train_data, 
        model_param["batch_size"], shuffle=True, num_workers=0, collate_fn=collate_fn)

    # load validation data
    val_data = ehr.EHR("dataset/EHR","val")
    val_data_loader = DataLoader(val_data, 
        model_param["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    # use data model to update model_param
    data_model_param = parse_data_model(train_data)
    model_param.update(data_model_param)
    use_gpu = model_param["use_gpu"]

    # init model
    model = NeuMF(**model_param)
    early_stopper = EarlyStopping(patience=model_param["early_stop"], larger_better=True)

    if model_param["use_gpu"]:
        model.cuda()

    print("Model Inited.")
    optimizer = torch.optim.Adam(model.parameters(),lr=model_param["lr"],weight_decay=0)


    for epoch in range(model_param["num_epoch"]):
        total_loss = 0
        model.train()

        for idx, (feat, dise) in enumerate(train_data_loader):
            pred, pred_neg, emb_user, emb_dise, neg_emb_dise = model.forward(feat, dise)

            bpr_loss = create_bpr_loss(pred, pred_neg)

            l2_loss = create_l2_loss(emb_user, emb_dise, neg_emb_dise)

            loss = bpr_loss + model_param["weight_decay"]*l2_loss
  
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += bpr_loss.item()
        
        print("{} Epoch {}/{}: train loss: {:.6f}".format(now(),epoch+1,
            model_param["num_epoch"],total_loss))

        # do evaluation on recall and ndcg
        metric_result, eval_log, eval_result  = evaluate(model, val_data_loader, [5])
        print("{} Epoch {}/{}: [Val] {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))

        early_stopper(metric_result["ndcg_5"], model, "neumf")

        if early_stopper.early_stop:
            print("[Early Stop] {} Epoch {}/{}: {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))
            break

    # eval on test set
    # load test data
    test_data = ehr.EHR("dataset/EHR","test")
    test_data_loader = DataLoader(test_data, 
        model_param["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    test_metric, test_log, test_result =  evaluate(model, test_data_loader, top_k_list=[1,3,5,10])
    print("[Test] {}: {}".format(now(), test_log))
    print("Training Done.")

def create_bpr_loss(pred, pred_neg):
    maxi = torch.sigmoid(pred - pred_neg).log()
    bpr_loss = -torch.mean(maxi)

    # use softplus loss
    # bpr_loss = torch.sum(torch.nn.functional.softplus(-(pred-pred_neg)))

    return bpr_loss

def create_l2_loss(emb_user, emb_dise, neg_emb_dise):
    batch_size = emb_user.shape[0]
    l2_loss = torch.sum(emb_user**2) + torch.sum(emb_dise**2) + torch.sum(neg_emb_dise**2)
    l2_loss = l2_loss / batch_size
    return l2_loss

def evaluate(model, data_loader, top_k_list=[3,5]):
    model.eval()
    result_map = defaultdict(list)

    # get all disease embeddings
    emb_dise = model.gen_all_dise_emb()

    # let's compute Recall@K and NDCG@K;
    for idx, (feat,dise) in enumerate(data_loader):
        with torch.no_grad():
            emb_user = model.forward_user(feat)
            batch_pred_score = model.forward_user_dise(emb_user, emb_dise)

        for j,dise_ in enumerate(dise):
            # evluate top k
            for top_k in top_k_list:
                pred_top_k = torch.argsort(-batch_pred_score, 1)[:,:top_k] + 1
            
                rank_rel = (pred_top_k[j] == dise_).float().detach().cpu().numpy()

                # count hit
                result_map["hit_{}".format(top_k)].append(rank_rel.sum())

                # count ndcg
                log2_iplus1 = np.log2(1+np.arange(1,top_k+1))

                # get dcg
                dcg_k = np.sum((2**(rank_rel) - 1) / log2_iplus1)

                # get idcg
                best_rank_rel = rank_rel[np.argsort(-rank_rel)]
                idcg_k = np.sum((2**best_rank_rel-1) / log2_iplus1)

                if idcg_k == 0:
                    ndcg_k = 0
                else:
                    ndcg_k = dcg_k / idcg_k

                result_map["ndcg_{}".format(top_k)].append(ndcg_k)

    # parse eval results
    final_result = {}
    print_log = ""
    for idx,top_k in enumerate(top_k_list):
        ndcg = np.mean(result_map["ndcg_{}".format(top_k)])
        recall = np.mean(result_map["hit_{}".format(top_k)])
        print_log += "Recall@{}: {:.4f}, nDCG@{}: {:.4f}.".format(top_k, recall, top_k, ndcg)
        final_result["ndcg_{}".format(top_k)] = ndcg
        final_result["recall_{}".format(top_k)] = recall

    return final_result, print_log, result_map


if __name__ == '__main__':
    import fire
    fire.Fire()
