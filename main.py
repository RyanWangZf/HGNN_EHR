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

from utils import evaluate

def train(**kwargs):

    setup_seed(2020)

    model_param = default_config()
    model_param = parse_kwargs(model_param, kwargs)

    # load hard maps
    if model_param["hard_ratio"] > 0:
        model_param["hard_map"] = np.load("dataset/hard_dise.npy", allow_pickle=True).item()
    
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

    # optimizer = torch.optim.Adam(gnn.parameters(),lr=model_param["lr"],weight_decay=model_param["weight_decay"])

    optimizer = torch.optim.Adam(gnn.parameters(),lr=model_param["lr"], weight_decay=0)


    # init sampler for netative sampling during training.
    dsd_sampler = DSD_sampler("dataset/EHR")
    print("D-S-D Sampler Inited.")

    for epoch in range(model_param["num_epoch"]):
        total_loss = 0
        gnn.train()

        for idx, (feat, dise) in enumerate(train_data_loader):
            pred, pred_neg, emb_user, emb_dise, neg_emb_dise = gnn.forward(feat, dise, dsd_sampler)
            
            bpr_loss = create_bpr_loss(pred, pred_neg)

            l2_loss = create_l2_loss(emb_user, emb_dise, neg_emb_dise)
            loss = bpr_loss + model_param["weight_decay"]*l2_loss
            # loss = bpr_loss

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            total_loss += bpr_loss.item()
            # print(idx,total_loss)
        
        print("{} Epoch {}/{}: train loss: {:.6f}".format(now(),epoch+1,
            model_param["num_epoch"],total_loss))

        # do evaluation on recall and ndcg

        metric_result, eval_log, eval_result  = evaluate(gnn, val_data_loader, dsd_sampler, [5])
        print("{} Epoch {}/{}: [Val] {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))

        early_stopper(metric_result["ndcg_5"], gnn, "gnn")

        if early_stopper.early_stop:
            print("[Early Stop] {} Epoch {}/{}: {}".format(now(),epoch+1, model_param["num_epoch"], eval_log))
            break

    # eval on test set
    # load test data
    test_data = ehr.EHR("dataset/EHR","test")
    test_data_loader = DataLoader(test_data, 
        model_param["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    test_metric, test_log, test_result =  evaluate(gnn, test_data_loader, dsd_sampler, top_k_list=[1, 3, 5, 10])
    print("[Test] {}: {}".format(now(), test_log))
    print("Training Done.")


def create_bpr_loss(pred, pred_neg):
    maxi = torch.sigmoid(pred - pred_neg).log()
    bpr_loss = -torch.mean(maxi)
    return bpr_loss

def create_l2_loss(emb_user, emb_dise, neg_emb_dise):
    batch_size = emb_user.shape[0]
    l2_loss = torch.sum(emb_user**2) + torch.sum(emb_dise**2) + torch.sum(neg_emb_dise**2)
    l2_loss = l2_loss / batch_size
    return l2_loss


def predict(model, data_loader):
    # TODO
    model.eval()
    pass

if __name__ == '__main__':
    import fire
    fire.Fire()
