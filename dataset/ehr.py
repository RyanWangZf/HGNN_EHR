# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from collections import defaultdict

import pdb
import re, os

np.random.seed(2020)

class EHR:
    def __init__(self, prefix="EHR", mode="train"):
        self.prefix = prefix
        assert mode in ["train", "test", "val"]
        path = os.path.join(prefix, "{}/data.npy".format(mode))

        label_path = os.path.join(prefix, "{}/label.npy".format(mode))
        
        self.data = np.load(path,allow_pickle=True).item()

        self._parse_metapath_params(self.data)


        # disease type
        self.label = np.load(label_path,allow_pickle=True)
        
        self.num_sample = self.label.shape[0]

        for k in self.data.keys():
            assert self.data[k].shape[0] == self.num_sample

        # load maps
        self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy"),allow_pickle=True).item()
        self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy"),allow_pickle=True).item()
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
        self.regex_split = re.compile("[。，？；！、]")
        self.regex_eng = re.compile("[a-zA-Z\\\/\(\)]")
        self.regex_dp = re.compile("[.,?;!/)('。，？；！、]")

    def pre_processing(self):
        prefix = self.prefix
        self.dise2id = read_disease2id(prefix)

        # build neighborhood maps
        self.symp2id = {}
        self.id2symp = defaultdict(list)
 
        data_path = os.path.join(prefix,"sampleNew_p_202005111638.xlsx")
        self.raw_df = pd.read_excel(data_path,encoding="utf-8")
        self.raw_df["dises"] = self.raw_df["3"].apply(lambda x: x.split("|"))

        # rearange df by spliting the raw diseases
        f_out = open(os.path.join(prefix,"data.txt"),"w",encoding="utf-8")
        
        uid = 1
        for i in range(len(self.raw_df)):
            line = self.raw_df.iloc[i]
            num_result = line["num_result"]

            try:
                kw_list = self.get_keywords_raw(line["8"])
            except:
                print("[Warning] Cannot process symptoms {} at line {}".format(line["8"], i))

            kw_list = list(set(kw_list))

            # *******************************
            # The following executes the standardization system
            # try:
            #     symps = num_result.split("@@")
            # except:
            #     print("[Warning] Cannot process symptoms {} at line {}".format(num_result, i))
            #     continue

            # up_symp2id = {}
            # symp_ids = []
            # for symp in symps:
            #     idx, sym = symp.split(",")
            #     sym = self.regex_dp.sub("",sym).strip()
            #     idx = re.findall("[0-9]+",idx)[0]
            #     up_symp2id[sym] = idx
            #     self.id2symp[idx].append(sym)
            #     symp_ids.append(idx)

            # self.symp2id.update(up_symp2id)

            # # do symptom standardization
            # uq_symp_ids = list(set(symp_ids))
            # kw_list = [self.id2symp[u][0] for u in uq_symp_ids]
            # *******************************

            # for ds in line["dises"]:
            #     diseid = self.dise2id.get(ds)
            #     if diseid is None:
            #         continue
            #     wrt_line = str(uid) + "\t" + str(diseid) + "\t" + "\t".join(kw_list) + "\n"
            #     f_out.write(wrt_line)
            #     uid += 1

            # only select one disease
            uid = line[0]
            ds = line["dises"][0]
            diseid = self.dise2id.get(ds)
            if diseid is None:
                continue
            wrt_line = str(uid) + "\t" + str(diseid) + "\t" + "\t".join(kw_list) + "\n"
            f_out.write(wrt_line)
            # uid += 1

        f_out.close()
        print("Pre-processing done.")


    def split(self):
        prefix = self.prefix
        self.symp2id = {}
        self.dise2symp = defaultdict(list)
        self.symp2dise = defaultdict(list)
        self.symp2user = defaultdict(list)
        self.user2symp = defaultdict(list)

        # read pre-processed data
        f_in = open(os.path.join(self.prefix, "data.txt"),"r", encoding="utf-8")
        all_lines = f_in.readlines()
        f_in.close()

        all_idx = np.arange(len(all_lines))
        np.random.shuffle(all_idx)

        num_tr_sample = int(len(all_idx) * 0.8)
        num_va_sample = int(len(all_idx) * 0.1)

        tr_idx = all_idx[:num_tr_sample]
        va_idx = all_idx[num_tr_sample:num_tr_sample+num_va_sample]
        te_idx = all_idx[num_tr_sample+num_va_sample:]

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
        f_out_te_more_name = os.path.join(te_prefix,"data_moresymp.txt")
        f_out_te_more = open(f_out_te_more_name,"w",encoding="utf-8")

        f_out_va_name = os.path.join(va_prefix,"data.txt")
        f_out_va = open(f_out_va_name, "w", encoding="utf-8")

        num_kws = 1
        num_tr = 0
        print("Start processing train...")
        for idx in tr_idx:
            data = all_lines[idx]
            line_data = data.strip().split("\t")
            symp_list = line_data[2:]
            uid = line_data[0]
            diseid = line_data[1]
            kws_list = []
            for kw in symp_list:
                if self.symp2id.get(kw) is None:
                    self.symp2id[kw] = str(num_kws)
                    num_kws += 1
                sid = self.symp2id[kw]
                kws_list.append(sid)
                self.dise2symp[str(diseid)].append(str(sid))
                self.symp2user[str(sid)].append(str(uid))
                self.symp2dise[str(sid)].append(str(diseid))
                self.user2symp[str(uid)].append(str(sid))

            wrt_line = str(uid) + "\t" + diseid + "\t" + "\t".join(kws_list) + "\n"
            f_out_tr.write(wrt_line)
            num_tr += 1

        # unique dict's values
        self.unique_dict(self.dise2symp)
        self.unique_dict(self.user2symp)
        self.unique_dict(self.symp2dise)
        self.unique_dict(self.symp2user)

        f_out_tr.close()

        np.save(os.path.join(prefix,"dise2symp.npy"),self.dise2symp)
        np.save(os.path.join(prefix,"symp2dise.npy"),self.symp2dise)
        np.save(os.path.join(prefix,"symp2id.npy"),self.symp2id)
        np.save(os.path.join(prefix,"symp2user.npy"),self.symp2user)
        np.save(os.path.join(prefix,"user2symp.npy"),self.user2symp)

        print("Start processing validation & test...")
        num_va = 0
        for idx in va_idx:
            data = all_lines[idx]
            line_data = data.strip().split("\t")

            uid = line_data[0]
            diseid = line_data[1]
            symp_list = line_data[2:]

            kws_list = []
            for kw in symp_list:
                sid = self.symp2id.get(kw)
                if sid is None:
                    continue
                else:
                    kws_list.append(sid)

            if len(kws_list) == 0:
                continue

            wrt_line = str(uid) + "\t" + diseid + "\t" + "\t".join(kws_list) + "\n"
            f_out_va.write(wrt_line)
            num_va += 1

        f_out_va.close()

        num_te = 0
        for idx in te_idx:
            data = all_lines[idx]
            line_data = data.strip().split("\t")

            uid = line_data[0]
            diseid = line_data[1]
            symp_list = line_data[2:]

            kws_list = []
            for kw in symp_list:
                sid = self.symp2id.get(kw)
                if sid is None:
                    continue
                else:
                    kws_list.append(sid)

            if len(kws_list) == 0:
                continue

            wrt_line = str(uid) + "\t" + diseid + "\t" + "\t".join(kws_list) + "\n"
            f_out_te.write(wrt_line)
            num_te += 1

            if len(kws_list) > 3:
                f_out_te_more.write(wrt_line)


        f_out_te.close()

        print("# Tr:{}, # Va: {} # Te:{}".format(num_tr, num_va, num_te))
        print("# Symptoms Used:", len(self.symp2id.keys()))

    def post_processing(self, mode="train", use_pmi=True):
        # ---------------
        # hyper parameter setup

        "for metapath DSD"
        num_DSD_1_hop = 20
        num_DSD_2_hop = 5

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
        self.dise2symp = np.load(os.path.join(prefix,"dise2symp.npy"),allow_pickle=True).item()
        self.symp2user = np.load(os.path.join(prefix,"symp2user.npy"),allow_pickle=True).item()
        self.user2symp = np.load(os.path.join(prefix,"user2symp.npy"),allow_pickle=True).item()
        self.symp2dise = np.load(os.path.join(prefix,"symp2dise.npy"),allow_pickle=True).item()


        if use_pmi:
            from scipy import sparse
            self.pmi_ss_path = os.path.join(prefix, "pmi_ss_mat.npz")
            self.pmi_sd_path = os.path.join(prefix, "pmi_sd_mat.npz")
            self.symp2dise_pmi = sparse.load_npz(self.pmi_sd_path)
            # build pmi_ds_mat
            self.dise2symp_pmi = sparse.csr_matrix(self.symp2dise_pmi.T.todense())
            self.symp_count_path = os.path.join(prefix, "sympcount.npy")
            self.dise_count_path = os.path.join(prefix, "disecount.npy")
            self.disecount = np.load(self.dise_count_path, allow_pickle=True).item()
            self.sympcount = np.load(self.symp_count_path, allow_pickle=True).item()
            c_ar, i_ar = [], []
            for k,v in self.sympcount.items():
                c_ar.append(v)
                i_ar.append(int(k))
            sympcount_mat = sparse.csr_matrix((c_ar, (i_ar, [0]*len(i_ar))))
            self.sympcount_ar = sympcount_mat.toarray().flatten()
            self.num_all_symp = self.sympcount_ar.sum()

            # build dise count array
            c_ar, i_ar = [], []
            for k,v in self.disecount.items():
                c_ar.append(v)
                i_ar.append(int(k))
            disecount_mat = sparse.csr_matrix((c_ar, (i_ar, [0]*len(i_ar))))
            self.disecount_ar = disecount_mat.toarray().flatten()
            self.num_all_dise = self.disecount_ar.sum()
            self.dise2symp_s_neighbor = {}


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
            if use_pmi:
                if self.dise2symp_s_neighbor.get(dise) is None:
                    ds_scores = self.dise2symp_pmi[dise]
                    ds_score_vals = ds_scores.data
                    ds_score_inds = ds_scores.indices
                    p_dise = self.disecount_ar[dise] / self.num_all_dise
                    p_symp = self.sympcount_ar[ds_score_inds] / self.num_all_symp
                    pmi_ds = np.log2((1e-8 + ds_score_vals/self.num_all_dise)/(1e-8 + p_dise * p_symp))
                    Z = - np.log2((ds_score_vals +1e-8)/(1e-8+ self.num_all_dise))
                    npmi_ds = pmi_ds / Z
                    sort_idx = np.argsort(-npmi_ds)
                    dsd_1_hop_nb = ds_score_inds[sort_idx][:100]
                    self.dise2symp_s_neighbor[dise] = dsd_1_hop_nb
                else:
                    dsd_1_hop_nb = self.dise2symp_s_neighbor[dise]

                dsd_1_hop_nb = self.random_select(dsd_1_hop_nb, num_DSD_1_hop)

            else:
                # random sampling
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

        if use_pmi:
            dise2symp_s_neighbor_path = os.path.join(prefix,"dise2symp_s_n.npy")
            np.save(dise2symp_s_neighbor_path, self.dise2symp_s_neighbor)

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

    def get_keywords(self, raw_symp):
        symps = raw_symp.split("@@")
        dic = {}
        for symp in symps:
            idx, sym = symp.split(",")
            sym = self.regex_dp.sub("",sym).strip()
            idx = re.findall("[0-9]+",idx)[0]
            dic[sym] = idx

        return dic

    def get_keywords_raw(self, raw_symp):
        raw_kw_list = self.regex_split.sub("##", raw_symp).strip().split("##")

        kw_list = []
        for x in raw_kw_list:
            kw = self.regex_eng.sub("",x)
            if len(kw) > 0:
                kw_list.append(kw)
        return kw_list

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

    def build_pmi_matrix(self, prefix="EHR"):
        from scipy import sparse
        # get num_symp
        self.symp2dise_dict = np.load(os.path.join(prefix,"symp2dise.npy"),allow_pickle=True).item()
        self.num_symp = len(self.symp2dise_dict.keys())

        symp2symp = defaultdict(list)
        symp2dise = defaultdict(list)
        symp2count = defaultdict(int)
        dise2count = defaultdict(int)

        self.pmi_ss_path = os.path.join(prefix, "pmi_ss_mat.npz")
        self.pmi_sd_path = os.path.join(prefix, "pmi_sd_mat.npz")
        self.symp_count_path = os.path.join(prefix, "sympcount.npy")
        self.dise_count_path = os.path.join(prefix, "disecount.npy")
        self.dise2symp_path = os.path.join(prefix, "dise2symp.npy")

        pmi_datapath = os.path.join(prefix, "train/data.txt")
        fin = open(pmi_datapath, "r", encoding="utf-8")
        for line in fin.readlines():
            line_data = line.strip().split("\t")
            diseid = line_data[1]
            symps = line_data[2:]
            dise2count[diseid] += 1
            for symp in symps:
                symp2symp[symp].extend(symps)
                symp2count[symp] += 1
                symp2dise[symp].append(diseid)

        self.sympcount = symp2count
        csr_data, csr_col, csr_row = [],[],[]
        csr_sd_data, csr_sd_col, csr_sd_row = [],[],[]
        for symp in range(1,self.num_symp):
            coc_count = pd.value_counts(symp2symp[str(symp)])
            csr_data.extend(coc_count.values.tolist())
            csr_row.extend([symp]*len(coc_count))
            csr_col.extend(coc_count.index.values.astype(int).tolist())

            coc_sd_count = pd.value_counts(symp2dise[str(symp)])
            csr_sd_data.extend(coc_sd_count.values.tolist())
            csr_sd_row.extend([symp]*len(coc_sd_count))
            csr_sd_col.extend(coc_sd_count.index.values.astype(int).tolist())

        self.symp2symp = sparse.csr_matrix((csr_data,(csr_row,csr_col)))
        self.symp2dise = sparse.csr_matrix((csr_sd_data, (csr_sd_row, csr_sd_col)))
        self.disecount = dise2count
        sparse.save_npz(self.pmi_ss_path, self.symp2symp)
        sparse.save_npz(self.pmi_sd_path, self.symp2dise)
        np.save(self.symp_count_path ,self.sympcount)
        np.save(self.dise_count_path, self.disecount)
        print("Done PMI Mat Building.")


def read_disease2id(prefix="EHR"):
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
    return disease2id

if __name__ == '__main__':
    ehr = EHR_load(prefix = "EHR")
    ehr.pre_processing()
    ehr.split()
    ehr.build_pmi_matrix("EHR")
    ehr.post_processing("train")
    ehr.post_processing("test")
    ehr.post_processing("val")