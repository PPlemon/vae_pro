import argparse
from molecules.model import MoleculeVAE
import os
import lzma
import re
import random
import zlib
import pandas as pd
import struct
import pickle
import h5py
import base64

from rdkit import Chem
import numpy as np
from functools import reduce
from molecules.utils import load_dataset, decode_smiles_from_indexes
from sklearn.model_selection import train_test_split
base64_charset = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base32_charset = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7']
# m = Chem.MolFromMolFile('data/substances.mol2')
# print(Chem.MolToSmiles(m))
# def get_arguments():
#     parser = argparse.ArgumentParser(description='Prepare data for training')
#     parser.add_argument('infile', type=str, help='Input file name')
#     parser.add_argument('outfile', type=str, help='Output file name')
#
#     return parser.parse_args()

# def tosmiles():
# data_zlib = []
# logp_zlib = []
# data = []
# inf = open('data/tox21_10k_data_all.sdf', 'rb')
# fsuppl = Chem.ForwardSDMolSupplier(inf)
# mols = [x for x in fsuppl]
# for mol in mols:
#     if mol is None: continue
#     c = Chem.MolToSmiles(mol)
#     data.append(c)
# print(len(mols))
# # return data

# data = pd.read_hdf('data/zinc-1.h5', 'table')
# smiles = data['structure'].map(lambda x: str(x.ljust(120)))
# smiles = data['smiles']
# logp = data['logp']
# qed = data['qed'][:]
# sas = data['sas'][:]
# train_idx, test_idx = map(np.array,
#                               train_test_split(smiles.index, test_size = 0.20))
# print(smiles[test_idx])
# data_train = smiles[train_idx]
# data_test = smiles[test_idx]
# property_train = logp[train_idx]
# property_test = logp[test_idx]


# for i in train_idx:
#     num = []
#     smile = smiles[i].encode()
#     compressed = zlib.compress(smile)
#     string = ['%02x' % b for b in compressed]
#     c = ''.join(string)
#     a = int(c, 16)
#     s = str(a).ljust(150)
#     print(len(s))
#     for j in s:
#         num.append(j)
#     filename = 'zlib_train.txt'
#     with open(filename, 'a') as f:
#         f.write(str(num) + "\n")
#     filename = 'property_train.txt'
#     with open(filename, 'a') as f:
#         f.write(str(logp[i]) + "\n")
#
# for i in test_idx:
#     num = []
#     smile = smiles[i].encode()
#     compressed = zlib.compress(smile)
#     string = ['%02x' % b for b in compressed]
#     c = ''.join(string)
#     a = int(c, 16)
#     s = str(a).ljust(150)
#     print(len(s))
#     for j in s:
#         num.append(j)
#     filename = 'zlib_test.txt'
#     with open(filename, 'a') as f:
#         f.write(str(num) + "\n")
#     filename = 'property_test.txt'
#     with open(filename, 'a') as f:
#         f.write(str(logp[i]) + "\n")
def zl(smile):
    num = []
    smiles = smile.encode()
    compressed = zlib.compress(smiles).ljust(64)
    while compressed:
        s = struct.unpack('h', compressed[-2:])[0]
        compressed = compressed[:-2]
        num.append(s)
        print(s)
    return num
def zs(smile):
    num = []
    smiles = smile.encode().ljust(32)
    # compressed = zlib.compress(smiles).ljust(72)
    while smiles:
        s = struct.unpack('h', smiles[-2:])[0]/20000
        smiles = smiles[:-2]
        num.append(s)
        print(s)
    return num

# charset = list(reduce(lambda x, y: set(y) | x, smiles, set()))
# print(charset)
# for smile in smiles:
#     smiles = smile.encode()
#     compressed = zlib.compress(smiles)
#     vec = [0] * len(charset)
#     for c in compressed:
#         for index, value in enumerate(charset):
#             if c == value:
#                 vec[index] += 1
#     print(vec)
#     break
# print(charset)
# print(len(charset))

def zlibcharset(smiles):
    compresseds = []
    for smile in smiles:
        smile = smile.ljust(120)
        compressed = zlib.compress(smile.encode())
        compresseds.append(compressed)
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset
def zlibvector(smile, charset):
    smile = smile.ljust(120)
    compressed = zlib.compress(smile.encode())
    vec = [0] * len(charset)
    num = []
    for c in compressed:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec
def lzmacharset(smiles):
    compresseds = []
    for smile in smiles:
        smile = smile.ljust(120)
        compressed = lzma.compress(smile.encode())
        compresseds.append(compressed)
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset
def lzmavector(smile, charset):
    smile = smile.ljust(120)
    compressed = lzma.compress(smile.encode())
    vec = [0] * len(charset)
    num = []
    for c in compressed:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec
def basecharset(smiles):
    compresseds = []
    for smile in smiles:
        smile = smile.ljust(120)
        compressed = base64.b64encode(smile.encode())
        compresseds.append(compressed)
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset

# def basevector(smile, charset):
#     smile = smile.ljust(120)
#     compressed = base64.b64encode(smile.encode())
#     vec = [0] * len(charset)
#     num = []
#     for c in compressed:
#         for index, value in enumerate(charset):
#             if c == value:
#                 vec[index] += 1
#     return vec



def lzw(string):
    dictionary = {chr(i): i for i in range(1, 128)}
    last = 129
    p = ""
    result = []
    for c in string:
        pc = p + c
        if pc in dictionary:
            p = pc
        else:
            result.append(dictionary[p])
            dictionary[pc] = last
            last += 1
            p = c
    if p != '':
        result.append(dictionary[p])
    return result
def lzwcharset(smiles):
    compresseds = []
    for smile in smiles:
        compresseds.append(lzw(smile))
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset
def lzwvector(smile,charset):
    compressed = lzw(smile)
    vec = [0] * len(charset)
    for c in compressed:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec

# charset = lzwcharset(smiles)
# print(charset)
# print(len(charset))
type = 0
# for smile in smiles[train_idx]:
#     num = [0]*120
#     i = 0
#     str = ''
#     chr = []
#     compressed = zlib.compress(smile.encode())
#     print(compressed)
#     print(zlib.decompress(compressed))
#     while compressed:
#         s = struct.unpack('B', compressed[-1:])[0]
#         compressed = compressed[:-1]
#         num[i] = s
#         i += 1
#     print(num)
#     for c in range(len(num)):
#         if num[c] == num[c+1] == 0:
#             break
#         s1 = struct.pack('i', num[c])
#         print(num[c])
#         print(s1)
#     print(str)
#         # print(repr(s))
#         # d = zlib.decompress(s1)
#         # print(d)
#     type += 1
#     if type == 1:
#         break

# base64解码
# for smile in smiles[train_idx]:
#     num = []
#     x = ''
#     s = bytes(x, "ascii")
#     # smile = smile.ljust(120)
#     print(len(smile))
#     compressed = base64.b64encode(smile.encode())
#     print(compressed)
#     print(len(compressed))
#     compressed = compressed.ljust(150)
#     print(base64.b64decode(compressed))
#     for i in compressed:
#         # print(i)
#         num.append(i)
#     # while num:
#     #     s = base64.b64decode(str(num[c]).encode())
#     for c in range(len(num)):
#         if num[c] == num[c + 1] == 32:
#             break
#         # print(chr(num[c]))
#         s += chr(num[c]).encode()
#     # if len(s) % 3 == 1:
#     #     s += bytes("==", "ascii")
#     # elif len(s) % 3 == 2:
#     #     s += bytes("=", "ascii")
#     # print(len(s))
#     sm = base64.b64decode(s)
#
#     print(bytes.decode(sm))
#     type += 1
#     if type == 10:
#         break
SMILES_CHARS = [' ',
                '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']
smiles_chars = ['S', '\\', 'B', 'P', '-', ']', 'C', '@', '+', 'l', 'n', 'c', '(', '5', '8', 's', 'O', 'I', '7', '6', 'H', '1', 'F', 'o', '/', '3', ')', '2', '=', 'r', '4', '#', 'N', '[']
# def basedecoder(num):
#     x = ''
#     s = bytes(x, "ascii")
#     for c in range(len(num)):
#         if num[c] == num[c + 1] == 32:
#             break
#             # print(chr(num[c]))
#         s += chr(num[c]).encode()
#     print(s)
#     if len(s) % 3 == 1:
#         s += bytes("==", "ascii")
#     elif len(s) % 3 == 2:
#         s += bytes("=", "ascii")
#     # if len(s) % 4 == 1:
#     #     s += bytes("===", "ascii")
#     # elif len(s) % 4 == 2:
#     #     s += bytes("==", "ascii")
#     # elif len(s) % 4 == 3:
#     #     s += bytes("=", "ascii")
#     print(len(s))
#     s = base64.b64decode(s)
#     for j in s:
#         if chr(j) not in smiles_chars:
#             print('解码失败')
#             s = base64.b64decode(base64.b64encode(SOURCE.encode()))
#             break
#     sm = bytes.decode(s)
#     print(sm)
#     return sm
# def interpolate(source, dest, steps):
#     step = (dest - source)/steps
#     results = []
#     for i in range(steps):
#         item = source + (step * i)
#         item = list(map(int, item))
#         for n in range(len(item)):
#             if item[n] == item[n + 1] == 32:
#                 break
#             elif item[n] > 122:
#                 item[n] = 122
#             elif item[n] == 61:
#                 if item[n+1] == item[n+2] == 32:
#                     continue
#                 else:
#                     item[n] = 57
#             else:
#                 while not re.match('[A-Za-z0-9+/]', chr(item[n])):
#                     item[n] += 1
#         sampled = basedecoder(item)
#         results.append((i, item, sampled))
#     return results
# def interpolate(source, dest, steps):
#     source = list(source)
#     dest = list(dest)
#     step4 = (dest[3] - source[3]) / steps
#     item = source
#     results = []
#     for i in range(steps):
#         item[3] = int(item[3] + (step4 * i))
#         for n in range(len(item)):
#             if item[n] == item[n + 1] == 32:
#                 break
#             elif item[n] > 122:
#                 item[n] = 122
#             elif item[n] == 61:
#                 if item[n+1] == item[n+2] == 32:
#                     continue
#                 else:
#                     item[n] = 57
#             else:
#                 while not re.match('[A-Za-z0-9+/]', chr(item[n])):
#                     item[n] += 1
#         sampled = basedecoder(item)
#         results.append((i, item, sampled))
#     return results
def pp_decoder(item):
    s = ''
    for n in range(len(item)):
        if item[n] == item[n + 1] == 0:
            break
        else:
            for key in smiles_dictionary:
                if smiles_dictionary[key] == item[n]:
                    s += key
    return s
def interpolate(source, dest, steps):
    step = (dest - source) / steps
    results = []
    for i in range(steps):
        item = source + (step * i)
        item = list(map(int, item))
        sampled = pp_decoder(item)
        results.append((i, item, sampled))
    return results
base64_dictionary = {
                  'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'J': 9, 'K': 10, 'L': 11,
                  'M': 12, 'N': 13, 'O': 14, 'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22,
                  'X': 23, 'Y': 24, 'Z': 25,
                  'a': 26, 'b': 27, 'c': 28, 'd': 29, 'e': 30, 'f': 31, 'g': 32, 'h': 33, 'i': 34, 'j': 35, 'k': 36,
                  'l': 37, 'm': 38, 'n': 39, 'o': 40, 'p': 41, 'q': 42, 'r': 43, 's': 44, 't': 45, 'u': 46, 'v': 47,
                  'w': 48, 'x': 49, 'y': 50, 'z': 51,
                  '0': 52, '1': 53, '2': 54, '3': 55, '4': 56, '5': 57, '6': 58, '7': 59, '8': 60, '9': 61, '+': 62,
                  '/': 63}
def basevector(smile):
    num = []
    smile = smile.ljust(120)
    compressed = base64.b64encode(smile.encode())
    for c in compressed:
        i = base64_dictionary[chr(c)]/100
        num.append(i)
    return num
# SOURCE = 'N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1'
# # DEST = 'c1cc(cc(c1)Cl)NNC(=O)c2cc(cnc2)Br'
# DEST = 'Cc1ccnc(c1)NC(=O)Cc2cccc3c2cccc3'
# dest = np.array(basevector(DEST))
# steps = 20
# for smile in smiles:
#     source = np.array(basevector(smile))
#     results = interpolate(source, dest, steps)
#     for result in results:
#         print(result[0], result[2])
smiles_dictionary = {
                  'C': 1,
                  'c': 2,
                  'B': 3,
                  'P': 4,
                  '-': 5,
                  ']': 6,
                  'S': 7,
                  '@': 8,
                  '+': 9,
                  'l': 10,
                  'n': 11,
                  '(': 12,
                  '5': 13,
                  '8': 14,
                  's': 15,
                  'O': 16,
                  'I': 17,
                  '7': 18,
                  '6': 19,
                  'H': 20,
                  '1': 21,
                  'F': 22,
                  'o': 23,
                  '/': 24,
                  '3': 25,
                  ')': 26,
                  '2': 27,
                  '=': 28,
                  'r': 29,
                  '4': 30,
                  '#': 31,
                  'N': 32,
                  '[': 33,
                  '\\': 34}
def pp(smile):
    result = [0]*120
    i = 0
    for c in smile:
        result[i] = smiles_dictionary[c]
        i += 1
    return result
# for smile in smiles:
#     print(pp(smile))
#     break


# data = pd.read_hdf('smiles-big.h5', 'table')
# keys = data['structure'].map(len) < 121
# data = data[keys]
# smiles = data['structure'][:]
# print(len(data))
# compresseds = []
# for smile in smiles:
#     smiles_vector = []
#     smile = smile.ljust(120)
#     compressed = base64.b64encode(smile.encode())
#     print(compressed)
#     for c in compressed:
#         smiles_vector.append(chr(c))
#     compresseds.append(smiles_vector)
# charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
# print(len(charset))
# print(charset)

# length = 500000
# data = pd.read_hdf('data/smiles_500k.h5', 'table')
# keys = data['structure'].map(len) < 121
# if length <= len(keys):
#     data = data[keys].sample(n = length)
# else:
#     data = data[keys]
# smiles = data['structure'].map(lambda x: list(x.ljust(120)))
# charset = list(reduce(lambda x, y: set(y) | x, smiles, set()))
# filename = 'sousuo.txt'
# with open(filename, 'a') as f:
#     f.write(str(charset) + "\n")

# DEST = 'c1cc(cc(c1)Cl)NNC(=O)c2cc(cnc2)Br'
# print(base64.b64encode(DEST.encode()))

# SOURCE = 'N#Cc1ccc(-c2ccc(O[C@@H](C(=O)N3CCCC3)c3ccccc3)cc2)cc1'
# # DEST = 'c1cc(cc(c1)Cl)NNC(=O)c2cc(cnc2)Br'
# DEST = 'Cc1ccnc(c1)NC(=O)Cc2cccc3c2cccc3'
# source = np.array(vector(SOURCE))
# dest = np.array(vector(DEST))
# print(source)
# print(dest)
# steps = 200
# results = interpolate(source, dest, steps)
# for result in results:
#     print(result[0], result[2])




# n_topics = 20
# train = []
# test = []
# for i in smiles[train_idx]:
#     train.append(basevector(i, charset))
#     break
# for j in smiles[test_idx]:
#     test.append(basevector(j, charset))
#     break


# isomap降维
# from isomap import isomap
# D = np.array([[1, 2, 3, 4], [2, 1, 5, 6], [3, 5, 1, 7], [4, 6, 7, 1]])
# data_train = isomap(D, 2, 3)
# data_test = isomap(D, 2, 3)
# print(data_train)
# print(data_test)
# from sklearn.manifold import Isomap
# embedding = Isomap(n_components=100)
# train_transformed = embedding.fit_transform(np.array(train))
# print(train_transformed)

# lle降维
# f = open(r"Smiles.txt")
# lines = f.readlines()
# charset = zlibcharset(lines)
# for i in lines:
#     train.append(zlibvector(i, charset))
# from sklearn.manifold import LocallyLinearEmbedding
# Y = LocallyLinearEmbedding(n_neighbors=20, n_components=50).fit_transform(train)
# print(Y)

# tsne降维
# from sklearn.manifold import TSNE
# tsne = TSNE(n_components=2, random_state=0)
# t = tsne.fit_transform(np.array(train))
# print(t)


# kpca降维
# from sklearn.decomposition import KernelPCA
# kpca = KernelPCA(kernel='rbf', n_components=2)
# newMat = kpca.fit_transform(D)
# print(newMat)



# lda降维
# ldamodeltrain = lda.LDA(n_topics = n_topics, n_iter = 10, random_state = 1) #初始化模型, n_iter迭代次数
# ldamodeltrain.fit(data_train)
# data_train = np.array(ldamodeltrain.components_[:])
# ldamodeltest = lda.LDA(n_topics = n_topics, n_iter = 10, random_state = 1) #初始化模型, n_iter迭代次数
# ldamodeltest.fit(data_test)
# data_test = np.array(ldamodeltest.components_[:])
# print(data_train[0])
# print(ldamodeltrain.doc_topic_[:2])


# print(len(max(smiles)))
    # string = ['%02x' % b for b in compressed]
    # c = ''.join(string)
    # a = int(c, 16)
    # s = str(a)
    # s = s.ljust(150)
    # for i in s:
    #     num.append(i)
    # data_zlib.append(num)


# def z(st):
#     num = []
#     smiles = st.encode()
#     compressed = zlib.compress(smiles)
#     string = ['%02x' % b for b in compressed]
#     c = ''.join(string)
#     a = int(c, 16)
#     s = str(a).ljust(150)
#     for i in s:
#         num.append(i)
#     return num

# fw = open('dataFile.txt', 'wb')
# pickle.dump(data_zlib, fw, -1)
# pickle.dump(logp, fw, -1)
# fw.close()
# def chunk_iterator(dataset, chunk_size=1000):
#     chunk_indices = np.array_split(np.arange(len(dataset)),
#                                     len(dataset)/chunk_size)
#     for chunk_ixs in chunk_indices:
#         chunk = dataset[chunk_ixs]
#         yield (chunk_ixs, chunk)
#     raise StopIteration


# data_train, data_test, charset1 = load_dataset('data/processed_25000.h5')
# charset = []
# for i in range(len(charset1)):
#     charset.append(bytes.decode(charset1[i]))
# print(charset)
# print(len(charset))



# h5f.close()
# h5f = h5py.File('zlib.h5', 'a')
# for z in data_zlib:
#     h5f.create_dataset('data_zlib', data = z)
# h5f.close()

# h5转txt
# h5f = h5py.File('data/per_all_base64_250000.h5', 'r')
# # smiles_train = h5f['smiles_train'][:]
# smiles_test = h5f['smiles_test'][:]
# # charset = h5f['charset'][:]
# for i in range(len(smiles_test)):
#     item = smiles_test[i].argmax(axis=1)
#     print(item)
#     sampled = decode_smiles_from_indexes(item, base64_charset)
#     sampled = sampled.replace(' ', '').encode()
#     sampled = base64.b64decode(sampled)
#     print(sampled)
#     filename = 'Smiles_250000_test.txt'
#     with open(filename, 'a') as f:
#         f.write(sampled + "\n")

# txt转h5

import numpy
from molecules.util import base32_vector, base64_vector, vector1, index_vector
# f0 = open(r"Smiles_25000_train.txt")
# f1 = open(r"Smiles_25000_test.txt")
# lines0 = f0.readlines()
# lines1 = f1.readlines()
# train = []
# test = []
# for s in lines0:
#     train.append(base32_vector(s))
# for s in lines1:
#     test.append(base32_vector(s))
# print(train[0])
# data_train = np.array(train)
# data_test = np.array(test)
# h5f = h5py.File('data/processed_base32_25000_new.h5', 'w')
# h5f.create_dataset('data_train', data=data_train)
# h5f.create_dataset('data_test', data=data_test)
# h5f.close()

# txt转预处理后带属性的h5
smiles_train = []
smiles_test = []
logp_train = []
logp_test = []
qed_train = []
qed_test = []
sas_train = []
sas_test = []
f0 = open(r"Smiles_25000_train.txt")
f1 = open(r"Smiles_25000_test.txt")
data = pd.read_hdf('data/zinc-1.h5', 'table')
smiles = data['smiles'][:]
logp = data['logp'][:]
qed = data['qed'][:]
sas = data['sas'][:]
lines0 = f0.readlines()
lines1 = f1.readlines()
for s0 in lines0:
    s0 = s0.replace('\n', '')
    for index, value in enumerate(smiles):
        if s0 == value:
            smiles_train.append(base64_vector(s0))
            logp_train.append(logp[index])
            qed_train.append(qed[index])
            sas_train.append(sas[index])
for s1 in lines1:
    s1 = s1.replace('\n', '')
    for index, value in enumerate(smiles):
        if s1 == value:
            smiles_test.append(base64_vector(s1))
            logp_test.append(logp[index])
            qed_test.append(qed[index])
            sas_test.append(sas[index])
smiles_train = np.array(smiles_train)
smiles_test = np.array(smiles_test)
logp_train = np.array(logp_train)
logp_test = np.array(logp_test)
qed_train = np.array(qed_train)
qed_test = np.array(qed_test)
sas_train = np.array(sas_train)
sas_test = np.array(sas_test)
h5f = h5py.File('data/per_all_base64_25000(64).h5', 'w')
h5f.create_dataset('smiles_train', data=smiles_train)
h5f.create_dataset('smiles_test', data=smiles_test)
h5f.create_dataset('logp_train', data=logp_train)
h5f.create_dataset('logp_test', data=logp_test)
h5f.create_dataset('qed_train', data=qed_train)
h5f.create_dataset('qed_test', data=qed_test)
h5f.create_dataset('sas_train', data=sas_train)
h5f.create_dataset('sas_test', data=sas_test)
# h5f.create_dataset('charset', data=charset1)
h5f.close()


# h5f = h5py.File('data/processed_25000.h5', 'r')
# smiles_train = h5f['smiles_train'][:]
# smiles_test = h5f['smiles_test'][:]
# logp_train = h5f['logp_train'][:]
# logp_test = h5f['logp_test'][:]
# qed_train = h5f['qed_train'][:]
# qed_test = h5f['qed_test'][:]
# sas_train = h5f['sas_train'][:]
# sas_test = h5f['sas_test'][:]
# charset = h5f['charset']
# print(charset)
# model = MoleculeVAE()
# if os.path.isfile('data/vae_model_zinc_2.h5'):
#     model.load(charset, 'data/vae_model_zinc_2.h5', latent_rep_size=196)
# else:
#     raise ValueError("Model file %s doesn't exist" % 'data/vae_model_zinc_2.h5')
# smiles_train_latent = model.encoder.predict(smiles_train)
# smiles_test_latent = model.encoder.predict(smiles_test)
# print(smiles_train_latent[0])
# h5f = h5py.File('data/per_all_latent_250000.h5', 'w')
# h5f.create_dataset('smiles_train_latent', data=smiles_train_latent)
# h5f.create_dataset('smiles_test_latent', data=smiles_test_latent)
# h5f.create_dataset('logp_train', data=logp_train)
# h5f.create_dataset('logp_test', data=logp_test)
# h5f.create_dataset('qed_train', data=qed_train)
# h5f.create_dataset('qed_test', data=qed_test)
# h5f.create_dataset('sas_train', data=sas_train)
# h5f.create_dataset('sas_test', data=sas_test)
# h5f.close()

#
# smiles_train = []
# smiles_test = []
# data = pd.read_hdf('data/zinc-1.h5', 'table')
# smiles = data['smiles']
# logp = data['logp']
# qed = data['qed']
# sas = data['sas']
# structures = smiles.map(lambda x: list(x.ljust(120)))
# charset1 = list(reduce(lambda x, y: set(y) | x, structures, set()))
# print(charset1)
# charset = []
# for i in charset1:
#     charset.append(i.encode())
# print(charset)
# train_idx, test_idx = map(np.array, train_test_split(smiles.index, test_size=0.10))
# for s0 in smiles[:224509]:
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
# # h5f.create_dataset('charset', data=charset)
# h5f.close()


# data = pd.read_hdf('data/zinc-1.h5', 'table')
# smiles = data['smiles'][:]
# logp = data['logp'][:]
# qed = data['qed'][:]
# sas = data['sas'][:]
# train_idx, test_idx = map(np.array, train_test_split(smiles.index, test_size=0.20))
# model = MoleculeVAE()
# if os.path.isfile('vae_model_base64_196_new.h5'):
#     model.load(base64_charset, 'vae_model_base64_196_new.h5', latent_rep_size=196)
# else:
#     raise ValueError("Model file %s doesn't exist" % 'vae_model_base64_196_new.h5')
# smiles_train_latent = model.encoder.predict(smiles[train_idx])
# smiles_test_latent = model.encoder.predict(smiles[test_idx])
# h5f = h5py.File('data/per_all_base64_latent_25000.h5', 'w')
# h5f.create_dataset('smiles_train_latent', data=smiles_train_latent)
# h5f.create_dataset('smiles_test_latent', data=smiles_test_latent)
# h5f.create_dataset('logp_train', data=logp[train_idx])
# h5f.create_dataset('logp_test', data=logp[test_idx])
# h5f.create_dataset('qed_train', data=logp[train_idx])
# h5f.create_dataset('qed_test', data=logp[test_idx])
# h5f.create_dataset('sas_train', data=logp[train_idx])
# h5f.create_dataset('sas_test', data=logp[test_idx])
# h5f.close()