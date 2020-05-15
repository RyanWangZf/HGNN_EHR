# -*- coding: utf-8 -*-

import numpy as np
import pdb
import os
from collections import defaultdict
import time
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, larger_better=True):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """

        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

        self.val_metric_max = 0
        self.val_metric_min = np.Inf

        self.delta = delta
        self.best_model = None

        "the metric is better if it is larger, e.g., auc."
        "if set False, it is smaller it is better, e.g. loss."
        self.larger_better = larger_better

    def __call__(self, val_metric, model, model_name=None):
        if self.larger_better:
            self.call_larger_better(val_metric, model,model_name)
        else:
            self.call_smaller_better(val_metric,model,model_name)


    def call_larger_better(self, val_metric, model, model_name=None):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, model_name)

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                self.load_checkpoint(model, model_name)
        else:
            self.best_score = score
            self.save_checkpoint(score, model, model_name)
            self.counter = 0

    def call_smaller_better(self, val_metric, model, model_name=None):
        score = val_metric

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model, model_name)
        elif score > self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                self.load_checkpoint(model, model_name)
        else:
            self.best_score = score
            self.save_checkpoint(score, model, model_name)
            self.counter = 0
        pass


    def save_checkpoint(self, val_metric, model, model_name=None):
        '''Saves model when validation loss decrease.'''
        if not os.path.exists("ckpt"):
            os.mkdir("ckpt")

        if model_name is None:
            torch.save(model.state_dict(), './ckpt/checkpoint.pt')
        else:
            torch.save(model.state_dict(), './ckpt/{}.pt'.format(model_name))

        if self.larger_better:
            self.val_metric_max = val_metric        
        else:
            self.val_metric_min = val_metric

    def load_checkpoint(self, model, model_name=None):
        if model_name is None:
            ckpt = torch.load("./ckpt/checkpoint.pt")
        else:
            ckpt = torch.load("./ckpt/{}.pt".format(model_name))

        model.load_state_dict(ckpt)

def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def collate_fn(batch):
    data, label = zip(*batch)
    return data, np.array(label)


def parse_data_model(data_model):
    param = {}
    param["num_symp"] = data_model.num_symp
    param["num_dise"] = data_model.num_dise
    for k,v in data_model.metapath_param.items():
        param[k] = v
    return param

def parse_kwargs(param, kwargs):

    for k in kwargs.keys():
        if k in ["symp_embedding_dim", "dise_embedding_dim", "layer_size_dsd", "layer_size_usu","embedding_dim"]:
            """Cope with embedding dimensions.
            """
            dim = kwargs[k]
            param["symp_embedding_dim"] = dim
            param["symp_embedding_dim"] = dim
            param["layer_size_dsd"] = [dim] * 2
            param["layer_size_usu"] = [dim] * 3

        else:
            param[k] = kwargs[k]

    print('*************************************************')
    print("User Config:")

    # print configurations
    for k in param.keys():
        print("{} => {}".format(k, param[k]))

    print('*************************************************')

    return param


def load_ckpt(path, model, use_gpu=True):
    if use_gpu:
        ckpt = torch.load(path)
    else:
        ckpt = torch.load(path, map_location=torch.device("cpu"))

    model.load_state_dict(ckpt)
    print("Load Checkpoint from", path)
    return


def read_dise2id(prefix="./dataset/EHR"):
    filename = os.path.join(prefix,"id2disease.txt")
    f = open(filename, "r", encoding="utf-8")
    data = f.readlines()
    disease2id = {}
    for i,line in enumerate(data):
        # id_d, ds = line.split("\t")
        ds = line.strip()
        ds_list = ds.split("#")
        for d in ds_list:
            disease2id[d.strip()] = i

    f.close()

    # build id2disease
    id2disease = {}
    for k,v in disease2id.items():
        id2disease[v] = k

    return disease2id, id2disease


def read_dise2id_deprecated(prefix="./dataset/EHR"):
    filename = os.path.join(prefix,"id2disease.txt")
    f = open(filename, "r", encoding="utf-8")
    data = f.readlines()
    disease2id = {}

    for i,line in enumerate(data):
        id_d, ds = line.split("\t")
        ds_list = ds.split("#")
        for d in ds_list:
            disease2id[d.strip()] = id_d

    f.close()

    # build id2disease
    id2disease = {}
    for k,v in disease2id.items():
        id2disease[v] = k

    return disease2id, id2disease


def read_symp2id(prefix="./dataset/EHR"):
    filename = os.path.join(prefix,"symp2id.npy")
    symp2id = np.load(filename,allow_pickle=True).item()
    id2symp = defaultdict()

    for k,v in symp2id.items():
        id2symp[v] = k

    return symp2id, id2symp

def parse_rank(pred_rank, id2dise):
    pred_list = []
    if len(pred_rank.shape) == 1:
        pred_rank = np.expand_dims(pred_rank,1)
        num_dim = 1
    else:
        num_dim = pred_rank.shape[1]

    for i in range(len(pred_rank)):
        this_rank = pred_rank[i]
        if num_dim > 1:
            pred_list.append([id2dise[x] for x in this_rank])
        else:
            pred_list.append(id2dise[this_rank[0]])

    pred_list = np.array(pred_list)

    return pred_list

def load_embedding(ckptpath="ckpt/gnn.pt", norm=True):
    res = torch.load(ckptpath, map_location=torch.device("cpu"))
    symp_emb = res["symp_embeds.weight"].cpu().numpy()
    dise_emb = res["dise_embeds.weight"].cpu().numpy()

    if norm:
        # need normalized
        symp_norm = np.linalg.norm(symp_emb, 2, 1)
        dise_norm = np.linalg.norm(dise_emb, 2, 1)

        symp_emb[1:] = symp_emb[1:] * np.expand_dims(1/symp_norm[1:],1)
        dise_emb[1:] = dise_emb[1:] * np.expand_dims(1/dise_norm[1:],1)

    return symp_emb, dise_emb

def parse_query(query):
    """Given inputs queries, parse them to be
    the format accepted by our GNN.
    """
    # TODO

    return
