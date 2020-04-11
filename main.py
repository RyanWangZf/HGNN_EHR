# -*- coding: utf-8 -*-
import time
import torch
import numpy as np
import pdb
from torch.utils.data import DataLoader
import torch.nn.functional as F

from collections import defaultdict

from dataset import ehr
from model import HGNN, DSD_sampler, USU_sampler
from utils import now, parse_kwargs, parse_data_model, EarlyStopping
from utils import setup_seed, collate_fn

from config import default_config

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
    gnn = HGNN(**model_param)
    early_stopper = EarlyStopping(patience=model_param["early_stop"], larger_better=True)

    if use_gpu:
        gnn.cuda()
    
    print("Model Inited.")

    optimizer = torch.optim.Adam(gnn.parameters(),
        lr=model_param["lr"],weight_decay=model_param["weight_decay"])

    # init sampler for netative sampling during training.
    dsd_sampler = DSD_sampler("dataset/EHR")
    print("D-S-D Sampler Inited.")

    for epoch in range(model_param["num_epoch"]):
        total_loss = 0
        gnn.train()

        for idx, (feat, dise) in enumerate(train_data_loader):
            pred, pred_neg = gnn.forward(feat, dise, dsd_sampler)
            
            loss = create_bpr_loss(pred, pred_neg)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += loss.item()
        
        print("{} Epoch {}/{}: train loss: {:.6f}".format(now(),epoch+1,
            model_param["num_epoch"],total_loss))

        # do evaluation on recall and ndcg

        metric_result, eval_log, eval_result  = evaluate(gnn, val_data_loader, dsd_sampler, [3,5])
        print("{} Epoch {}/{}: [Val] {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))

        early_stopper(metric_result["ndcg_5"], gnn)

        if early_stopper.early_stop:
            print("[Early Stop] {} Epoch {}/{}: {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))
            break

    # eval on test set
    # load test data
    test_data = ehr.EHR("dataset/EHR","test")
    test_data_loader = DataLoader(test_data, 
        model_param["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    test_metric, test_log, test_result =  evaluate(gnn, test_data_loader, dsd_sampler, top_k_list=[3,5])
    print("[Test] {}: {}".format(now(), test_log))
    print("Training Done.")


def create_bpr_loss(pred, pred_neg):
    maxi = torch.sigmoid(pred - pred_neg).log()
    bpr_loss = -torch.mean(maxi)
    return bpr_loss

def predict(model, data_loader):
    # TODO
    model.eval()
    pass

def evaluate(model, data_loader, dsd_sampler, top_k_list=[3,5]):
    """Evaluate the model's recall and nDCG.
    """

    model.eval()

    result_map = defaultdict(list)

    # forward to obtain the disease embeddings
    emb_dise = model.gen_all_dise_emb(dsd_sampler)

    # let's compute Recall@K and NDCG@K;
    for idx, (feat,dise) in enumerate(data_loader):
        data = model._array2dict(feat)
        with torch.no_grad():
            emb_user = model.forward_usu(data) # ?, k

            batch_pred_score = emb_user.matmul(emb_dise.T)
        
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
