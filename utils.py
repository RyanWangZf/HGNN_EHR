# -*- coding: utf-8 -*-

import numpy as np
import pdb
import os
from collections import defaultdict
import time

class DSD_sampler:
    """Given targeted array of diseases,
    sampling their neighborhood based D-S-D path.
    """
    def __init__(self, prefix):
        self.prefix = prefix
        # load maps
        self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy"),allow_pickle=True).item()
        self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy"),allow_pickle=True).item()
        self.num_dise = len(self.dise2symp.keys())
        self.num_symp = len(self.symp2dise.keys())

    def sampling(self, label, num_DSD_1_hop, num_DSD_2_hop):
        """Neighbor sampling for D-S-D metapath.
        """
        dise_ar = label

        data_dict = defaultdict(list)

        for l in range(len(dise_ar)):
            # ---------------
            # for DSD path
            dise = dise_ar[l]
            dsd_1_hop_nb = self.dise2symp[str(dise)]
            dsd_1_hop_nb = self.random_select(dsd_1_hop_nb, num_DSD_1_hop)
            dsd_1_hop_nb = self.fill_zero(dsd_1_hop_nb, num_DSD_1_hop)

            data_dict["dsd_1"].append(dsd_1_hop_nb)

            dsd_2_hop_nb_list = []

            for i in range(len(dsd_1_hop_nb)):
                symp = dsd_1_hop_nb[i]
                dsd_2_hop_nb = self.symp2dise.get(symp)
                if dsd_2_hop_nb is None:
                    dsd_2_hop_nb = []
                else:
                    dsd_2_hop_nb = self.random_select(dsd_2_hop_nb, num_DSD_2_hop)
                
                dsd_2_hop_nb = self.fill_zero(dsd_2_hop_nb, num_DSD_2_hop)

                data_dict["dsd_2_{}".format(i)].append(dsd_2_hop_nb)
            # ---------------
        
        for k in data_dict.keys():
            data_dict[k] = np.array(data_dict[k]).astype(int)

        return data_dict

    def fill_zero(self, ar, target_num):
        ar_len = len(ar)
        assert target_num >= ar_len
        if ar_len < target_num:
            ar = np.r_[ar, ["0"]*(target_num - ar_len)]
        return ar

    def random_select(self, ar, target_num):
        all_idx = np.arange(len(ar))
        np.random.shuffle(all_idx)
        return ar[all_idx[:target_num]]

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
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
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print("EarlyStopping counter: {} out of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                self.load_checkpoint(model)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        torch.save(model.state_dict(), './ckpt/checkpoint.pt')
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
        ckpt = torch.load("./ckpt/checkpoint.pt")
        model.load_state_dict(ckpt)


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


