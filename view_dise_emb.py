# -*- coding: utf-8 -*-
import os
import pdb

import torch
import numpy as np

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib.pyplot as plt

from utils import read_symp2id, read_dise2id

plt.style.use("bmh")

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

def load_embedding(ckptpath="ckpt/GNN.pt", norm=True):
    res = torch.load(ckptpath)
    symp_emb = res["symp_embeds.weight"].cpu().numpy()
    dise_emb = res["dise_embeds.weight"].cpu().numpy()

    if norm:
        # need normalized
        symp_norm = np.linalg.norm(symp_emb, 2, 1)
        dise_norm = np.linalg.norm(dise_emb, 2, 1)

        symp_emb[1:] = symp_emb[1:] * np.expand_dims(1/symp_norm[1:],1)
        dise_emb[1:] = dise_emb[1:] * np.expand_dims(1/dise_norm[1:],1)

    return symp_emb, dise_emb

if __name__ == '__main__':
    pca = PCA(n_components=2)
    # tsne = TSNE(n_components=2, random_state=2020)

    symp_emb, dise_emb = load_embedding(norm=True)
    dise2id, id2dise = read_dise2id()

    dise_emb_2d = pca.fit_transform(dise_emb[1:])
    # dise_emb_2d = (dise_emb_2d - dise_emb_2d.min(0)) / (dise_emb_2d.max(0)-dise_emb_2d.min(0))

    print(dise_emb_2d.shape)
    name_list = []
    fig,ax = plt.subplots()
    for i in range(dise_emb_2d.shape[0]):
        x,y = dise_emb_2d[i,0], dise_emb_2d[i,1]
        ax.scatter(dise_emb_2d[i,0], dise_emb_2d[i,1])
        name_raw = id2dise[str(i+1)]
        name = name_raw.split("|")[0]
        name_list.append(name)
        ax.annotate(name, (x,y))


    plt.show()

    
