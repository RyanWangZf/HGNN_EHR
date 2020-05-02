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

from sklearn.ensemble import GradientBoostingClassifier


def train(**kwargs):
    setup_seed(2020)

    model_param = default_config()
    model_param = parse_kwargs(model_param, kwargs)

    # load training data
    train_data = ehr.EHR("dataset/EHR","train")

    # load validation data
    val_data = ehr.EHR("dataset/EHR","val")

    # use data model to update model_param
    data_model_param = parse_data_model(train_data)
    model_param.update(data_model_param)

    # init model
    model = GradientBoostingClassifier(n_estimators=100,
        learning_rate=0.1,
        verbose=1,
        n_iter_no_change=10,
        random_state=10)

    train_feat, train_label = train_data.get_feat_data()

    print("Start Training.")
    model.fit(train_feat, train_label)

    print("Training Finished.")

    # eval on test set
    # load test data
    test_data = ehr.EHR("dataset/EHR","test")
    test_feat, test_label = test_data.get_feat_data()

    test_metric, test_log, test_result =  evaluate_clf(model, 
        test_feat, test_label,top_k_list=[3,5])

    print("[Test] {}: {}".format(now(), test_log))
    print("Training Done.")

def evaluate_clf(model, feat, label, top_k_list=[3,5]):
    result_map = defaultdict(list)

    pred_prob = model.predict_proba(feat)
    for top_k in top_k_list:
        pred_top_k = np.argsort(-pred_prob,1)[:, :top_k] + 1

        for i, target in enumerate(label):

            rank_rel = (pred_top_k[i] == target).astype(np.float)
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


def evaluate(model, data_loader, top_k_list=[3,5]):
    model.eval()
    result_map = defaultdict(list)

    # get all disease embeddings
    emb_dise = model.gen_all_dise_emb()

    # let's compute Recall@K and NDCG@K;
    for idx, (feat,dise) in enumerate(data_loader):
        with torch.no_grad():
            emb_user = model.forward_user(feat)

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
