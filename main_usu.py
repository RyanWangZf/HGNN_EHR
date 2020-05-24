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

class HGNN_USU(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(HGNN_USU, self).__init__()
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
        if kwargs.get("hard_map") is None or kwargs["hard_ratio"] == 0:
            self.hard_neg_map = {}
            for i in range(self.num_dise+1):
                self.hard_neg_map[i] = [i+1]
            self.hard_neg_map[i] = [i-1]
            self.mine_hard = False
        else:
            self.hard_neg_map = kwargs.get("hard_map")
            self.hard_ratio = kwargs["hard_ratio"]
            self.mine_hard = True

        # init embeddings
        self.symp_embeds = torch.nn.Embedding(
            self.num_symp+1,
            embed_dim,
            padding_idx=0,
            )

        self.dise_embeds = torch.nn.Embedding(
            self.num_dise+1,
            embed_dim,
            padding_idx=0,
            )

        # init weights based on meta-path params
        self.linear_usu_1 = torch.nn.Linear(
            embed_dim, 
            layer_size_usu[0],
            bias=False,)

    

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
                        
        # self.act_fn = torch.nn.LeakyReLU(0.2)
        self.act_fn = torch.nn.Tanh()
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
        # emb_s = F.normalize(emb_s, p=2, dim=1)

        emb_s_1_list = []

        for i in range(self.num_dsd_1_hop):
            dise_2_hop = data["dsd_2_{}".format(i)]
    
            # m(s-d)
            emb_d = self.dise_embeds(dise_2_hop) # ?, 2, k
            # emb_d = F.normalize(emb_d, p=2, dim=1)
            
            emb_s_ = emb_s[:,i]

            m_s_d = self.linear_dsd_2_1(emb_d) + \
                self.linear_dsd_2_2(emb_d.mul(torch.unsqueeze(emb_s_, 1))) # ?, 2, k

            m_s_s = self.linear_dsd_2_1(emb_s_) # ?,k
            m_s_s = self.dropout_fn(m_s_s)

            m_s_d = self._avg_on_real_neighbor(m_s_d, dise_2_hop) # ?,k
            m_s_d = self.dropout_fn(m_s_d)
            
            # cat
            # emb_s_1_ = self.act_fn(self.linear_cat_2(torch.cat([m_s_s, m_s_d], 1)))
            emb_s_1_ = self.act_fn(m_s_s + m_s_d)
            
            # emb_s_1_ = self.dropout_fn(emb_s_1_)
            emb_s_1_ = F.normalize(emb_s_1_, p=2, dim=1)

            emb_s_1_list.append(torch.unsqueeze(emb_s_1_,1))

        emb_s_1 = torch.cat(emb_s_1_list, 1) # ?, 10, k
        
        m_d_s = self.linear_dsd_1_1(emb_s_1) + \
            self.linear_dsd_1_2(emb_s_1.mul(torch.unsqueeze(target_emb_d,1))) # ?, 10, k
        
        symp_1_hop = data["dsd_1"]

        m_d_s = self._avg_on_real_neighbor(m_d_s, symp_1_hop) # ?, k
        m_d_d = self.linear_dsd_1_1(target_emb_d) # ?, k

        # emb_d_last = self.act_fn(self.linear_cat_1(torch.cat([m_d_d, m_d_s], 1)))
        emb_d_last = self.act_fn(m_d_s + m_d_d)
        
        # emb_d_last = self.dropout_fn(emb_d_last)
        # emb_d_last = F.normalize(emb_d_last, p=2, dim=1)

        return emb_d_last

    def forward_usu(self, data):
        emb_s_1 = self.symp_embeds(data["usu_1"]) #?, 5, k
        m_u_s = self.linear_usu_1(emb_s_1)
        symp_1_hop = data["usu_1"]

        m_u_s = self._avg_on_real_neighbor(m_u_s, symp_1_hop)
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

    def load_symp_embed(self, w2v_path="./ckpt/w2v"):
        """Load pretrained symptom embeddings from Word2Vec model.
        """
        from gensim.models import Word2Vec
        embed_dim = self.embed_dim

        w2v_model = Word2Vec.load(w2v_path)
        # init embedding matrix
        w2v_list = [[0]*embed_dim]
        for i in range(1,self.num_symp+1):
            w2v_list.append(w2v_model.wv[str(i)])

        w2v_param = torch.FloatTensor(w2v_list)
        self.symp_embeds.weight.data.copy_(w2v_param)
        # freeze the symptom embeddings
        self.symp_embeds.requires_grad = False
        print("Load pretrained symptom embeddings from", w2v_path)

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
        weight[weight >= 1e8] = 0
        
        return msg.sum(1).mul(weight)

    def _neg_sampling_dise(self, label):
        """Inputs:
        label: array of diseases
        
        Outputs:
        neg_dise: array of negative sampling diseases

        """
        num_sample = len(label)
        neg_dise = np.random.randint(1, self.num_dise+1, num_sample)

        if self.mine_hard is True:
            # mine hard example using hard map
            num_hard = int(num_sample * self.hard_ratio)
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            hard_idx = all_idx[:num_hard]
            for idx in hard_idx:
                neg_dise[idx] = np.random.choice(self.hard_neg_map[label[idx]])

        # remove duplicate
        same_idx = (label == neg_dise)
        if np.sum(same_idx) > 0:
            rep_neg_dise = [self.hard_neg_map[x][0] for x in label[same_idx]]
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
    gnn = HGNN_USU(**model_param)
    if kwargs["w2v"] is not None:
        # load w2v data
        gnn.load_symp_embed(kwargs["w2v"])
        
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

        early_stopper(metric_result["ndcg_5"], gnn, "gnn_usu")

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
