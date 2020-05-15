"""Usage: python eval.py main --ckpt=ckpt/GNN.pt --use_gpu=False
"""

# -*- coding: utf-8 -*-

import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader
import os
import pdb

from model import HGNN, DSD_sampler, USU_sampler
from dataset import ehr
from utils import parse_data_model, parse_kwargs, parse_rank
from utils import collate_fn, now, load_ckpt, read_dise2id

from config import default_config

def main(**kwargs):
    model_param = default_config()
    model_param.update({"top_k":3})

    model_param = parse_kwargs(model_param, kwargs)


    print("Start evaluating on top {} predictions.".format(model_param["top_k"]))

    # load map
    dise2id, id2dise = read_dise2id("dataset/EHR")

    # load train data model
    data_model = ehr.EHR("dataset/EHR","train")

    test_data = ehr.EHR("dataset/EHR","test")
    test_data_loader  = DataLoader(test_data, 
            model_param["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)

    data_model_param = parse_data_model(data_model)
    model_param.update(data_model_param)

    gnn = HGNN(**model_param)
    if model_param["use_gpu"]:
        gnn.cuda()

    ckpt_path = kwargs.get("ckpt")

    if ckpt_path is None:
        print("[Warning] Do not set ckpt path, load from the default path.")
        load_ckpt("ckpt/checkpoint.pt", gnn, model_param["use_gpu"])
    else:
        load_ckpt(ckpt_path, gnn, model_param["use_gpu"])

    dsd_sampler = DSD_sampler("dataset/EHR")
    usu_sampler = USU_sampler("dataset/EHR")

    gnn.eval()

    emb_dise = gnn.gen_all_dise_emb(dsd_sampler)

    rank_list = None
    dise_list = None

    for idx, (feat, dise) in enumerate(test_data_loader):

        this_dise_list = parse_rank(dise, id2dise)

        if dise_list is None:
            dise_list = this_dise_list
        else:
            dise_list = np.r_[dise_list, this_dise_list]

        # get symps
        symp_list = []
        for x in feat:
            symp_list.append(x["symp"])

        symp_ar = np.array(symp_list)

        # re-sampling users embeddings by their symptoms
        pred_rank = gnn.rank_query(symp_ar, emb_dise, usu_sampler, top_k=model_param["top_k"])

        # parse rank for print
        pred_list = parse_rank(pred_rank, id2dise)

        if rank_list is None:
            rank_list = pred_list
        else:
            rank_list = np.r_[rank_list, pred_list]

    # save results
    res_ar = np.c_[dise_list, rank_list]
    df_res = pd.DataFrame(res_ar)
    col_name = ["GroundTruth"] + ["Pred_"+str(i+1) for i in range(rank_list.shape[1])]
    df_res.columns = col_name
    df_res.to_csv("Test_Results.csv",encoding="utf-8")

    print("Test done, save results in", "Test_Results.csv")

if __name__ == '__main__':
    import fire
    fire.Fire()