# -*- coding: utf-8 -*-
"""python main_med2vec.py train --use_gpu=False
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import pdb, time, os

from gensim.models import Word2Vec
import gensim
from collections import defaultdict

from dataset import ehr
from utils import now, parse_kwargs, parse_data_model, EarlyStopping
from utils import setup_seed, collate_fn
from config import default_config


class MLP(torch.nn.Module):
    def __init__(self,*args,**kwargs):
        super(MLP, self).__init__()
        self.num_symp = kwargs["num_symp"]
        self.num_dise = kwargs["num_dise"]
        self.use_gpu = kwargs["use_gpu"]
        embed_dim = kwargs["symp_embedding_dim"]
        self.linear_1 = torch.nn.Linear(embed_dim, 64, bias=False)
        self.linear_2 = torch.nn.Linear(64, 64, bias=False)
        self.linear_3 = torch.nn.Linear(64, self.num_dise, bias=False)
        # self.linear = torch.nn.Linear(embed_dim, self.num_dise)
        # init embeddings
        self.symp_embeds = torch.nn.Embedding(
            self.num_symp+1,
            embed_dim,
            padding_idx=0,
            )

        self.w2v_model = kwargs["w2v_model"]
        # init embedding matrix
        w2v_list = [[0]*embed_dim]
        for i in range(1,self.num_symp+1):
            w2v_list.append(self.w2v_model.wv[str(i)])

        w2v_param = torch.FloatTensor(w2v_list)
        self.symp_embeds.weight.data.copy_(w2v_param)
        self.symp_embeds.requires_grad = False

    def forward(self, feat):
        data = self._array2dict(feat)
        emb_symp = self.symp_embeds(data["symp"])
        real_num_neighbor = torch.unsqueeze((data["symp"] != 0).sum(1).float(),1)
        weight = 1 / (real_num_neighbor+1e-8)
        weight[weight >= 1e8] = 0
        emb_user = emb_symp.sum(1).mul(weight)
        
        h = self.linear_1(emb_user)
        h = self.linear_2(h)
        pred = self.linear_3(h)
        # pred = self.linear(emb_user)
        return pred

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
    w2v_model_name = "./ckpt/w2v"

    if os.path.exists(w2v_model_name):
        print("load word2vec model from", w2v_model_name)
        # load model directly
        w2v_model = Word2Vec.load(w2v_model_name)

    else:
        # load data
        filename = "./dataset/EHR/train/data.txt"
        fin = open(filename, "r")
        corpus = []
        for line in fin.readlines():
            corpus.append(line.strip().split()[2:])

        # learn word2vec model
        start_time = time.time()
        w2v_model = Word2Vec(corpus, size=64, window=3, min_count=1, workers=4, sg=1)
        w2v_model.save("./ckpt/w2v")
        print("training done, costs {} secs.".format(time.time()-start_time))


    # start training and testing the MLP model
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


    # let's build a MLP for prediction
    model_param["w2v_model"] = w2v_model
    model = MLP(**model_param)

    early_stopper = EarlyStopping(patience=model_param["early_stop"], larger_better=True)

    if model_param["use_gpu"]:
        model.cuda()

    print("Model Inited.")
    optimizer = torch.optim.Adam(model.parameters(),lr=model_param["lr"],weight_decay=kwargs["weight_decay"])

    for epoch in range(model_param["num_epoch"]):
        total_loss = 0
        model.train()

        for idx, (feat, dise) in enumerate(train_data_loader):
            pred = model.forward(feat)

            if model_param["use_gpu"]:
                label = torch.LongTensor(dise).cuda()
            else:
                label = torch.LongTensor(dise)

            # label is [1,2,3...,27]
            loss = F.cross_entropy(pred, label-1)

            # multi-class xent loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        print("{} Epoch {}/{}: train loss: {:.6f}".format(now(),epoch+1,
            model_param["num_epoch"],total_loss))

        # do evaluation on recall and ndcg
        metric_result, eval_log, eval_result  = evaluate_clf(model, val_data_loader, [5])
        print("{} Epoch {}/{}: [Val] {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))

        early_stopper(metric_result["ndcg_5"], model, "med2vec")

        if early_stopper.early_stop:
            print("[Early Stop] {} Epoch {}/{}: {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))
            break

    # eval on test set
    # load test data
    test_data = ehr.EHR("dataset/EHR","test")
    test_data_loader = DataLoader(test_data, 
        model_param["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    test_metric, test_log, test_result =  evaluate_clf(model, test_data_loader, top_k_list=[1,3,5,10])
    print("[Test] {}: {}".format(now(), test_log))
    print("Training Done.")
    pass


def evaluate_clf(model, data_loader, top_k_list=[3,5]):
    model.eval()
    result_map = defaultdict(list)

    # let's compute Recall@K and NDCG@K;
    for idx, (feat,dise) in enumerate(data_loader):
        with torch.no_grad():
            batch_pred_score = model.forward(feat)

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