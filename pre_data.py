import argparse
from molecules.model import MoleculeVAE
import os
import re
import random
import pandas as pd
import struct
import pickle
import h5py
import base64
from rdkit import Chem
import numpy as np
from scipy.spatial.distance import pdist
from functools import reduce
from molecules.utils import load_dataset, decode_smiles_from_indexes
from sklearn.model_selection import train_test_split
base64_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base32_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7']


from molecules.util import base32_vector, base64_vector, vector1
# h5f = h5py.File('data/processed_25000.h5', 'r')
#
# charset1 = h5f['charset'][:]
# print(charset1)
# charset = []
# for i in charset1:
#     charset.append(i.decode())
# print(charset)
# data = pd.read_hdf('data/zinc-1.h5', 'table')
# smiles = data['smiles']
# logp = data['logp']
# qed = data['qed']
# sas = data['sas']
# smiles_data = []
# logp_data = []
# qed_data = []
# sas_data = []
# type = 0
# for index, value in enumerate(smiles):
#     for i in value:
#         if i not in charset:
#             type = 1
#             break
#     if type == 0:
#         smiles_data.append(value)
#         logp_data.append(logp[index])
#         qed_data.append(qed[index])
#         sas_data.append(sas[index])
#     type = 0
# smiles_data = np.array(smiles_data)
# logp_data = np.array(logp_data)
# qed_data = np.array(qed_data)
# sas_data = np.array(sas_data)
# output_smiles = open('smiles.pkl', 'wb')
# output_logp = open('logp.pkl', 'wb')
# output_qed = open('qed.pkl', 'wb')
# output_sas = open('sas.pkl', 'wb')
# output_charset = open('charset.pkl', 'wb')
# pickle.dump(smiles_data, output_smiles)
# pickle.dump(logp_data, output_logp)
# pickle.dump(qed_data, output_qed)
# pickle.dump(sas_data, output_sas)
# pickle.dump(charset, output_charset)
# output_smiles.close()
# output_logp.close()
# output_qed.close()
# output_sas.close()
# output_charset.close()




# 按8:1:1划分数据并编码
# smiles = open('smiles(40).pkl', 'rb')
# smiles = pickle.load(smiles)
# print(smiles[0])
# charset1 = list(reduce(lambda x, y: set(y) | x, smiles, set()))
# print(charset1)
# charset = []
# for i in charset1:
#     charset.append(i.encode())
# print(charset)
# logp = open('logp(40).pkl', 'rb')
# logp = pickle.load(logp)
# print(len(logp))
# qed = open('qed(40).pkl', 'rb')
# qed = pickle.load(qed)
# print(len(qed))
# sas = open('sas(40).pkl', 'rb')
# sas = pickle.load(sas)
# print(len(sas))
# # charset = open('charset.pkl', 'rb')
# # charset = pickle.load(charset)
# # print(len(charset))
# idx = int(len(smiles)/10)
# train_idx = 8*idx
# test_idx = 9*idx
# smiles_train = []
# smiles_val = []
# smiles_test = []
# for s0 in smiles[:train_idx]:
#     smiles_train.append(vector1(s0, charset1))
# for s1 in smiles[train_idx:test_idx]:
#     smiles_val.append(vector1(s1, charset1))
# for s2 in smiles[test_idx:]:
#     smiles_test.append(vector1(s2, charset1))
# h5f = h5py.File('data/per_all_40.h5', 'w')
# h5f.create_dataset('smiles_train', data=smiles_train)
# h5f.create_dataset('smiles_val', data=smiles_val)
# h5f.create_dataset('smiles_test', data=smiles_test)
# h5f.create_dataset('logp_train', data=logp[:train_idx])
# h5f.create_dataset('logp_val', data=logp[train_idx:test_idx])
# h5f.create_dataset('logp_test', data=logp[test_idx:])
# h5f.create_dataset('qed_train', data=qed[:train_idx])
# h5f.create_dataset('qed_val', data=qed[train_idx:test_idx])
# h5f.create_dataset('qed_test', data=qed[test_idx:])
# h5f.create_dataset('sas_train', data=sas[:train_idx])
# h5f.create_dataset('sas_val', data=qed[train_idx:test_idx])
# h5f.create_dataset('sas_test', data=sas[test_idx:])
# h5f.create_dataset('charset', data=charset)

# 按9:1划分数据并编码
# smiles = open('smiles(45).pkl', 'rb')
# smiles = pickle.load(smiles)
# print(smiles[0])
# # charset1 = list(reduce(lambda x, y: set(y) | x, smiles, set()))
# # charset1.insert(0, ' ')
# # print(charset1)
# # charset = []
# # for i in charset1:
# #     charset.append(i.encode())
# # print(charset)
# logp = open('logp(45).pkl', 'rb')
# logp = pickle.load(logp)
# print(len(logp))
# qed = open('qed(45).pkl', 'rb')
# qed = pickle.load(qed)
# print(len(qed))
# sas = open('sas(45).pkl', 'rb')
# sas = pickle.load(sas)
# print(len(sas))
# # charset = open('charset.pkl', 'rb')
# # charset = pickle.load(charset)
# # print(len(charset))
# idx = int(len(smiles)/10)
# train_idx = 9*idx
# smiles_train = []
# smiles_test = []
# for s0 in smiles[:train_idx]:
#     smiles_train.append(base64_vector(s0))
# for s1 in smiles[train_idx:]:
#     smiles_test.append(base64_vector(s1))
# h5f = h5py.File('data/per_all_base64_45(120).h5', 'w')
# h5f.create_dataset('smiles_train', data=smiles_train)
# h5f.create_dataset('smiles_test', data=smiles_test)
# h5f.create_dataset('logp_train', data=logp[:train_idx])
# h5f.create_dataset('logp_test', data=logp[train_idx:])
# h5f.create_dataset('qed_train', data=qed[:train_idx])
# h5f.create_dataset('qed_test', data=qed[train_idx:])
# h5f.create_dataset('sas_train', data=sas[:train_idx])
# h5f.create_dataset('sas_test', data=sas[train_idx:])
# h5f.create_dataset('charset', data=charset)

# vae压缩
# h5f = h5py.File('data/per_all_44(120).h5', 'r')
# smiles_train = h5f['smiles_train'][:]
# smiles_test = h5f['smiles_test'][:]
# logp_train = h5f['logp_train'][:]
# logp_test = h5f['logp_test'][:]
# qed_train = h5f['qed_train'][:]
# qed_test = h5f['qed_test'][:]
# sas_train = h5f['sas_train'][:]
# sas_test = h5f['sas_test'][:]
# charset = h5f['charset']
# # print(charset)
# model = MoleculeVAE()
# if os.path.isfile('data/vae_model_44(120).h5'):
#     model.load(charset, 'data/vae_model_44(120).h5', latent_rep_size=196)
# else:
#     raise ValueError("Model file %s doesn't exist" % 'data/vae_model_44(120).h5')
# smiles_train_latent = model.encoder.predict(smiles_train)
# smiles_test_latent = model.encoder.predict(smiles_test)
# print(smiles_train_latent[0])
# h5f = h5py.File('data/per_all_latent_44(120).h5', 'w')
# h5f.create_dataset('smiles_train_latent', data=smiles_train_latent)
# h5f.create_dataset('smiles_test_latent', data=smiles_test_latent)
# h5f.create_dataset('logp_train', data=logp_train)
# h5f.create_dataset('logp_test', data=logp_test)
# h5f.create_dataset('qed_train', data=qed_train)
# h5f.create_dataset('qed_test', data=qed_test)
# h5f.create_dataset('sas_train', data=sas_train)
# h5f.create_dataset('sas_test', data=sas_test)
# h5f.close()

# 数据编码
# smiles_train = []
# smiles_test = []
# data = pd.read_hdf('data/zinc-1.h5', 'table')
# smiles = data['smiles']
# logp = data['logp']
# qed = data['qed']
# sas = data['sas']
# for s0 in smiles[:224509]:
#     smiles_train.append(base32_vector(s0))
# for s1 in smiles[224509:]:
#     smiles_test.append(base32_vector(s1))
# smiles_train = np.array(smiles_train)
# smiles_test = np.array(smiles_test)
# print(smiles_test[0][0])
# h5f = h5py.File('data/per_all_base32_250000.h5', 'w')
# h5f.create_dataset('smiles_train', data=smiles_train)
# h5f.create_dataset('smiles_test', data=smiles_test)
# h5f.create_dataset('logp_train', data=logp[:224509])
# h5f.create_dataset('logp_test', data=logp[224509:])
# h5f.create_dataset('qed_train', data=qed[:224509])
# h5f.create_dataset('qed_test', data=qed[224509:])
# h5f.create_dataset('sas_train', data=sas[:224509])
# h5f.create_dataset('sas_test', data=sas[224509:])
# h5f.close()

# 平均欧式距离
# h5f = h5py.File('data/per_all_latent_test.h5', 'r')
# smiles_train_latent = h5f['smiles_train_latent'][:]
# print(len(smiles_train_latent))
# distances = []
# for i in range(10000):
#     x = random.randint(0, 10000)
#     source = smiles_train_latent[x]
#     y = random.randint(10000, 20000)
#     dest = smiles_train_latent[y]
#     Y = np.vstack([source, dest])  # 将x,y两个一维数组合并成一个2D数组 ；[[x1,x2,x3...],[y1,y2,y3...]]
#     distance = pdist(Y)
#     print(distance)
#     distances.append(distance)
# print(np.mean(distances))

# 字符分布
# h5f = h5py.File('data/per_all_250000.h5', 'r')
# smiles_train = h5f['smiles_train'][:]
# charset1 = h5f['charset'][:]
# charset = []
# for i in charset1:
#     charset.append(i.decode())
# print(charset)
# chr_num = [0 for i in range(len(base64_charset))]
# for s in smiles_train:
#     item = s.argmax(axis=1)
#     for i in item:
#         chr_num[i] += 1
#
# print(chr_num)

h5f = h5py.File('data/per_all_base64_45(120).h5', 'r')
smiles_train = h5f['smiles_train'][:]
# charset = h5f['charset'][:]
# print(charset)
print(len(smiles_train[0]), len(smiles_train[0][1]), smiles_train[0][60])

# 取固定长度字符
# smiles_data = []
# logp_data = []
# qed_data = []
# sas_data = []
# data = pd.read_hdf('data/zinc-1.h5', 'table')
# smiles = data['smiles']
# logp = data['logp']
# qed = data['qed']
# sas = data['sas']
# for index, value in enumerate(smiles):
#     if len(value) == 48:
#         smiles_data.append(value)
#         logp_data.append(logp[index])
#         qed_data.append(qed[index])
#         sas_data.append(sas[index])
# print(len(smiles_data))
# output_smiles = open('smiles(48).pkl', 'wb')
# output_logp = open('logp(48).pkl', 'wb')
# output_qed = open('qed(48).pkl', 'wb')
# output_sas = open('sas(48).pkl', 'wb')
# pickle.dump(smiles_data, output_smiles)
# pickle.dump(logp_data, output_logp)
# pickle.dump(qed_data, output_qed)
# pickle.dump(sas_data, output_sas)
# output_smiles.close()
# output_logp.close()
# output_qed.close()
# output_sas.close()


# 验证
# h5f = h5py.File('data/per_all_45(120).h5', 'r')
# data_train = h5f['smiles_train'][:]
# data_test = h5f['smiles_test'][:]
# print(len(data_train), len(data_test))
# charset = h5f['charset'][:]
# h5f.close()
# model = MoleculeVAE()
# if os.path.isfile('data/vae_model_45(120).h5'):
#     model.load(charset, 'data/vae_model_45(120).h5', latent_rep_size=196)
# else:
#     raise ValueError("Model file %s doesn't exist" % 'data/vae_base64_45(120).h5')
# data_test_vae = model.autoencoder.predict(data_test)
# count0 = 0
# count1 = 0
# for m in range(len(data_test_vae)):
#     item0 = data_test[m].argmax(axis=1)
#     item1 = data_test_vae[m].argmax(axis=1)
#     print(item0)
#     print(item1)
#     # break
#     for j in range(len(item0)):
#         if item0[j] == 0:
#             break
#         count0 += 1
#         if item0[j] != item1[j]:
#             count1 += 1
# print(count0, count1)
# print((count0-count1)/count0)


