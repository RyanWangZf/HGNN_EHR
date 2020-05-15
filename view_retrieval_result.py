import os
import pdb
import torch
import numpy as np

from utils import read_symp2id, read_dise2id, load_embedding

from EHR_retrieval import EHR_retrieval
from config import default_config
from model import HGNN, DSD_sampler, USU_sampler
from dataset import ehr
from utils import load_ckpt, parse_kwargs, parse_data_model

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
    datapath = os.path.join("dataset/EHR/test/data_moresymp.txt")
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

    for i, line in enumerate(lines):
        line_data = line.strip().split()
        uid = line_data[0]
        did = line_data[1]
        symps = line_data[2:]

        # select the first symptom and do inference
        init_symp = symps[0]
        symp_ar = np.array([init_symp])

        pred_rank = gnn.rank_query(symp_ar, emb_dise, usu_sampler, top_k=1)

        pdb.set_trace()
        pass

        ret_symps = ehr_ret(init_symp, param["top_k"])
        ret_symp_list = []
        for sid in ret_symps:
            ret_symp_list.append(id2symp[str(sid)])



if __name__ == '__main__':
    import fire
    fire.Fire()