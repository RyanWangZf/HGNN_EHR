# -*- coding: utf-8 -*-
import pdb
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from utils import read_dise2id

prefix = "./dataset/EHR"
filename = os.path.join(prefix, "data.txt")

dise2id, id2dise = read_dise2id()

fin = open(filename, "r", encoding="utf-8")

dise2symp = defaultdict(list)
symp2dise = defaultdict(list)
user2symp = defaultdict(list)
user2dise = defaultdict(list)

sympcount = defaultdict(int)
disecount = defaultdict(int)

num_user2symp = 0
for i, line in enumerate(fin.readlines()):
    line_data = line.strip().split()
    did = line_data[1]
    dise = id2dise[int(did)]
    symps = line_data[2:]

    if i % 10000 == 0:
        print("Processing line:", i)
    for symp in symps:
        dise2symp[dise].append(symp)
        symp2dise[symp].append(dise)
        sympcount[symp] += 1

    num_user2symp += len(np.unique(symps))
    disecount[dise] += 1

# nodes
num_user = i + 1
num_dise = len(dise2symp.keys())
num_symp = len(symp2dise.keys())

# links
num_user2dise = i + 1
num_dise2symp = 0
dise_list, dise_count = [],[]
for k in disecount.keys():
    dise_list.append(k)
    dise_count.append(disecount[k])
    d2s_list = dise2symp[k]
    d2s_list = np.unique(d2s_list)
    num_dise2symp += len(d2s_list)

df_disecount = pd.DataFrame({"dise":dise_list, "count":dise_count})
df_disecount = df_disecount.sort_values(by="count",ascending=False).reset_index(drop=True)
df_disecount.to_excel("disecount.xlsx")

symp_list, symp_count = [],[]
for k in sympcount.keys():
    symp_list.append(k)
    symp_count.append(sympcount[k])

df_sympcount = pd.DataFrame({"symp":symp_list, "count":symp_count})
df_sympcount = df_sympcount.sort_values(by="count",ascending=False).reset_index(drop=True)
df_sympcount.to_excel("sympcount.xlsx")

fin.close()

# print results
print("====>","RESULT","<====")
print("# user:", num_user)
print("# dise:", num_dise)
print("# symp:", num_symp)
print("# user2dise:", num_user2dise)
print("# user2symp:", num_user2symp)
print("# dise2symp:", num_dise2symp)
print("====>","RESULT","<====")

print("Done.")

