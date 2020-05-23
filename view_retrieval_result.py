"""python vie_retrieval_result.py main --mode="gpmi"
"""
import os
import pdb
import torch
import numpy as np
import pandas as pd

from collections import defaultdict

from utils import read_symp2id, read_dise2id, load_embedding

from EHR_retrieval import EHR_retrieval
from config import default_config
from model import HGNN, DSD_sampler, USU_sampler
from dataset import ehr
from utils import load_ckpt, parse_kwargs, parse_data_model

def calculate_rec_ndcg(pred_top_k, target, top_k, result_map):
    rank_rel = np.array(pred_top_k == int(target)).astype(float)
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


def build_result_log(result_map, top_k_list):
    # parse eval results
    final_result = {}
    print_log = ""
    for idx,top_k in enumerate(top_k_list):
        ndcg = np.mean(result_map["ndcg_{}".format(top_k)])
        recall = np.mean(result_map["hit_{}".format(top_k)])
        print_log += "Recall@{}: {:.4f}, nDCG@{}: {:.4f}.".format(top_k, recall, top_k, ndcg)
        final_result["ndcg_{}".format(top_k)] = ndcg
        final_result["recall_{}".format(top_k)] = recall

    return print_log


def main(**kwargs):
    # parse parameters
    param = default_config()
    param.update(
        {"mode":"sds",
        "top_k":10,
        "ckpt": "ckpt/GNN.pt",
        "use_gpu":False
        })

    param.update(kwargs)

    # read maps
    symp2id, id2symp = read_symp2id()
    dise2id, id2dise = read_dise2id()

    # read data
    datapath = os.path.join("dataset/EHR/test/data.txt")
    fin = open(datapath, "r", encoding="utf-8")
    lines = fin.readlines()

    data_model = ehr.EHR("dataset/EHR","train")

    # init retrieval system
    ehr_ret = EHR_retrieval(mode=param["mode"])

    # init and load model
    data_model_param = parse_data_model(data_model)
    param.update(data_model_param)
    param = parse_kwargs(param, kwargs)
    gnn = HGNN(**param)

    if param["use_gpu"]:
        gnn.cuda()

    ckpt_path = param.get("ckpt")
    if ckpt_path is None:
        print("[Warning] Do not set ckpt path, load from the default path.")
        load_ckpt("ckpt/checkpoint.pt", gnn, param["use_gpu"])
    else:
        load_ckpt(ckpt_path, gnn, param["use_gpu"])

    dsd_sampler = DSD_sampler("dataset/EHR")
    usu_sampler = USU_sampler("dataset/EHR")

    gnn.eval()

    emb_dise = gnn.gen_all_dise_emb(dsd_sampler)

    # init result list
    before_list = []
    after_list = []
    real_dise_list = []
    init_symp_list = []
    after_symp_list = []

    result_map_bfo = defaultdict(list)
    result_map_aft = defaultdict(list)
    # this is top_k for evaluation p@N, Rec@N, ...
    top_k_list = [1, 5]

    for i, line in enumerate(lines):
        line_data = line.strip().split()
        uid = line_data[0]
        did = line_data[1]
        real_dise_list.append(did)
        symps = line_data[2:]

        # select the first symptom and do inference
        init_symp = symps[0]
        init_symp_list.append(id2symp[init_symp])

        symp_ar = np.array([[init_symp]])

        pred_rank = gnn.rank_query(symp_ar, emb_dise, usu_sampler, top_k=5)

        # calculate statistics
        for top_k in top_k_list:
            pred_top_k = pred_rank[0][:top_k]
            calculate_rec_ndcg(pred_top_k, int(did), top_k, result_map_bfo)

        # print("true did:", did)
        # print("before:", pred_rank)
        before_list.append(pred_rank[0])

        rank_symp = ehr_ret(symp_idx=init_symp, top_k=param["top_k"])
        after_symp_list.append([id2symp[str(t)] for t in rank_symp])
        symp_ar = [np.concatenate([[init_symp], rank_symp], 0)]

        # symp_ar = np.array([symps])
        pred_rank = gnn.rank_query(symp_ar, emb_dise, usu_sampler, top_k=5)
        for top_k in top_k_list:
            pred_top_k = pred_rank[0][:top_k]
            calculate_rec_ndcg(pred_top_k, int(did), top_k, result_map_aft)

        # print("after:", pred_rank)
        after_list.append(pred_rank[0])

        ret_symps = ehr_ret(init_symp, param["top_k"])
        ret_symp_list = []
        for sid in ret_symps:
            ret_symp_list.append(id2symp[str(sid)])

        if i % 100 == 0:
            print("[line]:", i)

    # summary
    bf_log = build_result_log(result_map_bfo, top_k_list)
    af_log = build_result_log(result_map_aft, top_k_list)

    print("[before]: {}".format(bf_log))
    print("[after]: {}".format(af_log))

    # to result csv
    fout = open("retrieval_result_{}.txt".format(param["mode"]),"w", encoding="utf-8")
    fout.write("did\tbefore_pred\tafter_pred\tinit_symp\taftersymp\n")
    for i in range(len(init_symp_list)):
        wrtline = id2dise[int(real_dise_list[i])] + "\t" + id2dise[int(before_list[i][0])] + "\t" + id2dise[int(after_list[i][0])] + "\t" + init_symp_list[i] +"\t" + "#".join(after_symp_list[i]) + "\n"
        fout.write(wrtline)

    fin.close()
    fout.close()

    df_res = pd.read_table("retrieval_result_{}.txt".format(param["mode"]))
    df_res.to_excel("retrieval_result_{}.xlsx".format(param["mode"]), encoding="utf-8")
    print("Done")

if __name__ == '__main__':
    import fire
    fire.Fire()