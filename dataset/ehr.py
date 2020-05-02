# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict

import pdb
import os
import re
np.random.seed(2020)

class EHR:
    def __init__(self, prefix="EHR", mode="train"):
        self.prefix = prefix
        assert mode in ["train", "test", "val"]
        path = os.path.join(prefix, "{}/data.npy".format(mode))

        label_path = os.path.join(prefix, "{}/label.npy".format(mode))
        
        self.data = np.load(path).item()

        self._parse_metapath_params(self.data)


        # disease type
        self.label = np.load(label_path)
        
        self.num_sample = self.label.shape[0]

        for k in self.data.keys():
            assert self.data[k].shape[0] == self.num_sample

        # load maps
        self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy")).item()
        self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy")).item()
        self.num_dise = len(self.dise2symp.keys())
        self.num_symp = len(self.symp2dise.keys())

        print("Load # {} from {}.".format(self.num_sample,path))

    def _parse_metapath_params(self, data):
        self.metapath_param = {}

        # for DSD
        self.metapath_param["num_dsd_1_hop"] = self.data["dsd_1"].shape[1]
        self.metapath_param["num_dsd_2_hop"] = self.data["dsd_2_0"].shape[1]

        # for USU
        self.metapath_param["num_usu_1_hop"] = self.data["usu_1"].shape[1]
        self.metapath_param["num_usu_2_hop"] = self.data["usu_2_0"].shape[1]
        self.metapath_param["num_usu_3_hop"] = self.data["usu_3_0"].shape[1]

        return


    def get_feat_data(self):
        """Get array-like feature-label data,
        used for GBDT, MLP models.
        """
        feat = self.data["symp"]
        label = self.label
        return feat, label        

    def __getitem__(self, idx):
        assert idx < self.num_sample
        batch_map = defaultdict()
        for k in self.data.keys():
            batch_map[k] = self.data[k][idx]

        return batch_map, self.label[idx]

    def __len__(self):
        return self.num_sample


class EHR_load:
    def __init__(self, prefix="EHR"):
        self.prefix = prefix

    def pre_processing(self):
        prefix = self.prefix

        data_path = os.path.join(prefix,"sampleNew.csv")
        self.raw_df = pd.read_csv(data_path,encoding="utf-8")
        self.dise2id = read_disease2id(prefix)

        # build neighborhood maps
        self.symp2id = {}
        self.dise2symp = defaultdict(list)
        self.symp2dise = defaultdict(list)
        self.symp2user = defaultdict(list)
        self.user2symp = defaultdict(list)

        # split train and test
        all_idx = np.arange(len(self.raw_df))
        np.random.shuffle(all_idx)

        num_tr_sample = int(len(all_idx) * 0.7)
        num_va_sample = int(len(all_idx) * 0.1)

        self.df_tr = self.raw_df.iloc[all_idx[:num_tr_sample]]
        self.df_te = self.raw_df.iloc[all_idx[num_tr_sample:]]

        tr_prefix = os.path.join(prefix, "train")
        te_prefix = os.path.join(prefix, "test")
        va_prefix = os.path.join(prefix, "val")

        if not os.path.exists(tr_prefix):
            os.mkdir(tr_prefix)
        if not os.path.exists(te_prefix):
            os.mkdir(te_prefix)
        if not os.path.exists(va_prefix):
            os.mkdir(va_prefix)

        f_out_tr_name = os.path.join(tr_prefix,"data.txt")
        f_out_tr = open(f_out_tr_name, "w", encoding="utf-8")

        f_out_te_name = os.path.join(te_prefix,"data.txt")
        f_out_te = open(f_out_te_name, "w", encoding="utf-8")

        f_out_va_name = os.path.join(va_prefix,"data.txt")
        f_out_va = open(f_out_va_name, "w", encoding="utf-8")

        # start from 1
        # make 0 as NA
        num_kws = 1
        uid = 1

        print("Start Processing df train...")
        for idx in range(len(self.df_tr)):
            data = self.raw_df.iloc[idx]

            # get diseases
            diseid = self.dise2id.get(data["3"])
            if diseid is None:
                continue

            raw_symps = data["7"]
            kws = self.get_keywords(raw_symps)
            kws_list = []
            for kw in kws:
                if self.symp2id.get(kw) is None:
                    self.symp2id[kw] = str(num_kws)
                    num_kws += 1

                sid = self.symp2id[kw]
                kws_list.append(sid)

                self.dise2symp[str(diseid)].append(str(sid))
                self.symp2user[str(sid)].append(str(uid))
                self.symp2dise[str(sid)].append(str(diseid))
                self.user2symp[str(uid)].append(str(sid))


            # write line
            wrt_line = str(uid) + "\t" + diseid + "\t" + "\t".join(kws_list) + "\n"
            f_out_tr.write(wrt_line)
            uid += 1

        # unique dict's values
        self.unique_dict(self.dise2symp)
        self.unique_dict(self.user2symp)
        self.unique_dict(self.symp2dise)
        self.unique_dict(self.symp2user)

        # save maps
        num_tr_final = uid
        f_out_tr.close()
        np.save(os.path.join(prefix,"dise2symp.npy"),self.dise2symp)
        np.save(os.path.join(prefix,"symp2dise.npy"),self.symp2dise)
        np.save(os.path.join(prefix,"symp2id.npy"),self.symp2id)
        np.save(os.path.join(prefix,"symp2user.npy"),self.symp2user)
        np.save(os.path.join(prefix,"user2symp.npy"),self.user2symp)

        print("Start Processing df test...")

        for idx in range(len(self.df_te)):
            data = self.df_te.iloc[idx]

            # get diseases
            diseid = self.dise2id.get(data["3"])
            if diseid is None:
                continue

            raw_symps = data["7"]
            kws = self.get_keywords(raw_symps)
            kws_list = []

            for kw in kws:
                sid = self.symp2id.get(kw)
                if sid is None:
                    continue
                else:
                    kws_list.append(sid)

            if len(kws_list) == 0:
                continue

            uid += 1
            # write line
            wrt_line = str(uid) + "\t" + diseid + "\t" + "\t".join(kws_list) + "\n"
            f_out_te.write(wrt_line)

        f_out_te.close()
        num_te_final = uid - num_tr_final
        print("# Tr:{}, # Te:{}".format(num_tr_final,num_te_final))

        # split tr and va
        all_tr_idx = np.arange(num_tr_final)
        np.random.shuffle(all_tr_idx)

        f_tr = open(f_out_tr_name, "r", encoding="utf-8")
        lines = f_tr.readlines()

        for idx in all_tr_idx[:num_va_sample]:
            f_out_va.write(lines[idx])

        f_tr.close()
        f_out_va.close()

        print("Done Pre-Processing.")

    def post_processing(self, mode="train"):
        # ---------------
        # hyper parameter setup

        "for metapath DSD"
        num_DSD_1_hop = 10
        num_DSD_2_hop = 2

        "for metapath USU"
        num_USU_1_hop = 5
        num_USU_2_hop = 5
        num_USU_3_hop = 5
        # ---------------

        prefix = self.prefix
        assert mode in ["train","test","val"]

        data_prefix = os.path.join(prefix, mode)

        dise_ar, symp_ar = self.read_data(os.path.join(data_prefix,"data.txt"))
        num_dise = dise_ar.max() + 1

        # load maps
        self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy")).item()
        self.symp2user = np.load(os.path.join(prefix,"symp2user.npy")).item()
        self.user2symp = np.load(os.path.join(prefix,"user2symp.npy")).item()
        self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy")).item()

        print("start sampling neighborhood :", mode)

        # build maps
        data_dict = defaultdict(list)

        for l in range(len(dise_ar)):
            if l % 10000 == 0:
                print("Neigbor sampling line:",l)

            # ---------------
            # for DSD path
            dise = dise_ar[l]
            symp_list = symp_ar[l]

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

            # ---------------
            # for USU path
            usu_1_hop_nb = symp_list.astype(str)
            usu_1_hop_nb = self.random_select(usu_1_hop_nb, num_USU_1_hop)
            usu_1_hop_nb = self.fill_zero(usu_1_hop_nb, num_USU_1_hop)
            data_dict["usu_1"].append(usu_1_hop_nb)

            # user's symptoms is usu_1 !
            data_dict["symp"].append(usu_1_hop_nb)

            for i in range(len(usu_1_hop_nb)):
                symp = usu_1_hop_nb[i]
                usu_2_hop_nb = self.symp2user.get(symp)
                if usu_2_hop_nb is None:
                    usu_2_hop_nb = []
                else:
                    usu_2_hop_nb = self.random_select(usu_2_hop_nb, num_USU_2_hop)
                
                usu_2_hop_nb = self.fill_zero(usu_2_hop_nb, num_USU_2_hop)
                data_dict["usu_2_{}".format(i)].append(usu_2_hop_nb)

                for j in range(len(usu_2_hop_nb)):
                    uid = usu_2_hop_nb[j]
                    usu_3_hop_nb = self.user2symp.get(uid)
                    if usu_3_hop_nb is None:
                        usu_3_hop_nb = []
                    else:
                        usu_3_hop_nb = self.random_select(usu_3_hop_nb, num_USU_3_hop)

                    usu_3_hop_nb = self.fill_zero(usu_3_hop_nb, num_USU_3_hop)
                    data_dict["usu_3_{}".format(i*num_USU_2_hop+j)].append(usu_3_hop_nb)


        for k in data_dict.keys():
            data_dict[k] = np.array(data_dict[k]).astype(int)
        
        np.save(os.path.join(data_prefix,"data.npy"), data_dict)
        np.save(os.path.join(data_prefix,"label.npy"), dise_ar)

        print("Done :", mode)


    def read_data(self, path):
        # load data in array
        f_in = open(path,"r",encoding="utf-8")
        dise_list = []
        symp_list = []

        for line in f_in.readlines():
            line_data = line.split()
            dise_list.append(line_data[1])

            symp_list.append(np.array(line_data[2:]).astype("int"))

        symp_ar = np.array(symp_list)
        dise_ar = np.array(dise_list).astype("int")
        return dise_ar, symp_ar

    def get_keywords(self,sen):
        res = re.findall("(?<=\))[^\)]+(?=\()",sen)
        return res

    def unique_dict(self, dic):
        for k in dic.keys():
            values = dic[k]
            unq_val = np.unique(values)
            dic[k] = np.array(unq_val)
        return dic

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

def read_disease2id(prefix="EHR"):
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
    return disease2id

if __name__ == '__main__':
    ehr = EHR_load(prefix = "EHR")
    # ehr.pre_processing()
    ehr.post_processing("train")
    ehr.post_processing("test")
    ehr.post_processing("val")

