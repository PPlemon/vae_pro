import h5py
import numpy
import os
import random
import pandas
from keras import backend as K
from rdkit import Chem
from functools import reduce
import argparse
import matplotlib.pyplot as plt
from molecules.model import MoleculeVAE
from molecules.util import decode_smiles_from_indexes, vector1
from molecules.utils import load_dataset
from scipy.spatial.distance import pdist
#source = 'C=CCc1ccc(OCC(=O)N(CC)CC)c(OC)c1'
#dest = 'C=C(C)CNc1ccc([C@H](C)C(=O)O)cc1'
latent_dim = 196
steps = 100
width = 44
length = 500000
results = []

# 正态分布函数
def normfun(x, mu, sigma):
    pdf = numpy.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * numpy.sqrt(2 * numpy.pi))
    return pdf


def sampling(z_mean_, z_log_var_):  # 采样
    batch_size = K.shape(z_mean_)[0]  # 返回张量形状
    epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0., stddev=0.01)  # 噪声
    return z_mean_ + K.exp(z_log_var_ / 2) * epsilon

def interpolate_s(source, dest, charset, model, latent_dim):
    s = []

    source_encoded = vector1(source, charset)
    dest_encoded = vector1(dest, charset)
    # sampled0 = decode_smiles_from_indexes(numpy.array(source_encoded).argmax(axis=1)[0], charset)
    s.append(source_encoded)
    s.append(dest_encoded)
    s = numpy.array(s)
    latent = model.encoder.predict(s)
    source_x_latent = latent[0]
    sampled = model.decoder.predict(source_x_latent.reshape(1, latent_dim)).argmax(axis=2)[0]
    sampled = decode_smiles_from_indexes(sampled, charset)
    results.append(sampled)
    return results
def interpolate_q(source, dest, steps, charset, model, latent_dim):
    source = model.encoder.predict(source.reshape(1, width, len(charset)))[0]
    dest = model.encoder.predict(dest.reshape(1, width, len(charset)))[0]
    results = []
    for i in range(steps):
        # if i <= 0:
        #     return source
        # elif i >= 1:
        #     return dest
        # elif numpy.allclose(source, dest):
        #     return source
        # 求反余弦arccos   np.dot求矩阵乘法   np.linalg.norm求范数
        # 得到向量夹角
        # if 4 < i < 95:
        #     continue
        omega = numpy.arccos(numpy.dot(source / numpy.linalg.norm(source), dest / numpy.linalg.norm(dest)))
        # 求正弦
        so = numpy.sin(omega)
        item = numpy.sin((1.0 - i) * omega) / so * source + numpy.sin(i * omega) / so * dest
        sampled = model.decoder.predict(item.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        # print(sampled)
        results.append((i, sampled))
    return results
def interpolate_x(source, dest, steps, charset, model, latent_dim):
    source_x_latent = model.encoder.predict(source.reshape(1, width, len(charset)))
    dest_x_latent = model.encoder.predict(dest.reshape(1, width, len(charset)))
    step = (dest_x_latent - source_x_latent) / float(steps)
    results = []
    for i in range(steps):
        if 9 < i < 91:
            continue
        item = source_x_latent + (step * i)
        sampled = model.decoder.predict(item.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        results.append((i, sampled))
    return results

def main():
    type = 0
    t = 0
    s = []

    # data = pandas.read_hdf('data/processed_25000.h5', 'table')
    # keys = data['structure'].map(len) < 121
    # if length <= len(keys):
    #     data = data[keys].sample(n = length)
    # else:
    #     data = data[keys]
    # smiles = data['structure'].map(lambda x: list(x.ljust(120)))
    # charset = list(reduce(lambda x, y: set(y) | x, smiles, set()))
    # h5f = h5py.File('data/processed-big1.h5', 'r')
    # charset1 = h5f['charset'][:]
    # data_train, data_test, charset1 = load_dataset('data/processed_25000.h5')
    #

    h5f = h5py.File('data/per_all_44.h5', 'r')
    data_train = h5f['smiles_train'][:]
    data_test = h5f['smiles_test'][:]
    charset1 = h5f['charset'][:]
    charset = []
    for i in range(len(charset1)):
        charset.append(bytes.decode(charset1[i]))
    print(charset)
    # smiles_chars = [' ',
    #                 '#', '%', '(', ')', '+', '-', '.', '/',
    #                 '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    #                 '=', '@',
    #                 'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
    #                 'R', 'S', 'T', 'V', 'X', 'Z',
    #                 '[', '\\', ']',
    #                 'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
    #                 't', 'u']
    # charset = smiles_chars
    # charset.pop(0)
    # print(len(smiles_chars))
    print(len(charset))
    # print(charset)
    model = MoleculeVAE()
    if os.path.isfile('data/vae_model_44.h5'):
        model.load(charset, 'data/vae_model_44.h5', latent_rep_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % 'data/vae_model_44.h5')

    # 测试随机点
    # source = 'Cc1ccc(OC[C@H](O)CNC(C)(C)C)c2oc(=O)ccc12'
    # dest = 'C=CCc1ccc(OCC(=O)N(CC)CC)c(OC)c1'

    # 测试线性插值
    # for i in range(10):
    #     type = 0
    #     t = 0
    #     s = []
    #     x = random.randint(0, len(data_train)/2)
    #     source = data_train[x]
    #     y = random.randint(len(data_train)/2, len(data_train))
    #     dest = data_train[y]
    #     results = interpolate_x(source, dest, steps, charset, model, latent_dim)
    #     for result in results:
    #         m = Chem.MolFromSmiles(result[1])
    #         if m:
    #             type += 1
    #             # if str(result[2]) == str(source) or str(result[2]) == str(dest):
    #             #     print('repeat!')
    #             #     t += 1
    #             s.append(result[1])
    #             print(result[0], result[1])
    #     n = set(s)
    #     print(type, type / 1000, len(n))
    #     print(n)

    # 测试球型插值
    # for i in range(10):
    #     type = 0
    #     t = 0
    #     s = []
    #     x = random.randint(0, len(data_train)/2)
    #     source = data_train[x]
    #     y = random.randint(len(data_train)/2, len(data_train))
    #     dest = data_train[y]
    #     results = interpolate_q(source, dest, steps, charset, model, latent_dim)
    #     for result in results:
    #         m = Chem.MolFromSmiles(result[1])
    #         if m:
    #             type += 1
    #             # if str(result[2]) == str(source) or str(result[2]) == str(dest):
    #             #     print('repeat!')
    #             #     t += 1
    #             s.append(result[1])
    #             print(result[0], result[1])
    #     n = set(s)
    #     print(type, type / 100, len(n))
    #     print(n)

    # 测试种类数
    # for i in range(10):
    #     results = []
    #     type = 0
    #     t = 0
    #     s = []
    #     x = random.randint(0, len(data_train)-1)
    #     source = data_train[x]
    #     for j in range(1000):
    #         source_x_latent = model.autoencoder.predict(source.reshape(1, width, len(charset))).argmax(axis=2)[0]
    #         sampled = decode_smiles_from_indexes(source_x_latent, charset)
    #         results.append(sampled)
    #     for result in results:
    #         m = Chem.MolFromSmiles(result)
    #         if m:
    #             type += 1
    #             s.append(result)
    #             # print(result)
    #     n = set(s)
    #     print(type, type / 1000, len(n))
    #     print(n)

    # 画正态分布图
    # x_latent = model.encoder.predict(data_test)
    # latent = [[]for _ in range(latent_dim)]
    # for i in range(latent_dim):
    #     for j in range(len(data_test)):
    #         latent[i].append(x_latent[j][i])
    # # print(len(latent[0]), numpy.mean(latent[0]), numpy.var(latent[0]), numpy.std(latent[0]))
    # mean = numpy.mean(latent[0])
    # std = numpy.std(latent[0])
    # x = numpy.arange(-1, 1, 0.1)
    # # 设定 y 轴，载入刚才的正态分布函数
    # y = normfun(x, mean, std)
    # plt.plot(x, y)
    # plt.show()

    # 测试欧式距离
    # x = random.randint(0, len(data_train))
    # source = data_train[x]
    # y = random.randint(0, len(data_train))
    # dest = data_train[y]
    # source_x_latent0 = model.encoder.predict(source.reshape(1, width, len(charset)))
    # dest_x_latent0 = model.encoder.predict(source.reshape(1, width, len(charset)))
    # arr_mean = numpy.mean(dest_x_latent0)
    # arr_var = numpy.var(dest_x_latent0)
    # print(arr_mean, arr_var)
    # Y = numpy.vstack([source_x_latent0, dest_x_latent0])  # 将x,y两个一维数组合并成一个2D数组 ；[[x1,x2,x3...],[y1,y2,y3...]]
    # d0 = pdist(Y)
    # print(d0)

    # 1000*5解码率测试
    for i in range(1000):
        x = random.randint(0, len(data_test)-1)
        source = data_test[x]
        for m in range(5):
            source_x_latent = model.encoder.predict(source.reshape(1, width, len(charset)))
            # for j in range(100):
            sampled = model.decoder.predict(source_x_latent.reshape(1, latent_dim)).argmax(axis=2)[0]
            sampled = decode_smiles_from_indexes(sampled, charset)
            results.append(sampled)
            print(sampled)

    # 单分子平均欧式距离测试
    # x = random.randint(0, len(data_train))
    # source = data_train[x]
    # source_x_latent0 = model.encoder.predict(source.reshape(1, width, len(charset)))
    # for i in range(1000):
    #     source_x_latent = model.encoder.predict(source.reshape(1, width, len(charset)))
    #     Y = numpy.vstack([source_x_latent0, source_x_latent])  # 将x,y两个一维数组合并成一个2D数组 ；[[x1,x2,x3...],[y1,y2,y3...]]
    #     d0 = pdist(Y)
    #     s.append(d0)
    # print(numpy.mean(s))


    # vae性能测试
    # for i in range(1000):
    #     x = random.randint(0, len(data_train))
    #     source = data_train[x]
    #     sampled = model.autoencoder.predict(source.reshape(1, width, len(charset))).argmax(axis=2)[0]
    #     # print(sampled)
    #     # sampled0 = source.argmax(axis=1)
    #     # print(sampled0)
    #     # if str(sampled) == str(sampled0):
    #     #     type += 1
    #     sampled = decode_smiles_from_indexes(sampled, charset)
    #     results.append(sampled)
    #     print(sampled)

    # 测试解码率
    for result in results:
        m = Chem.MolFromSmiles(result)
        if m != None:
            type += 1
            s.append(result)
            print(result)
    n = set(s)
    print(type, len(results), len(n))

    # f = open(r"data/smiles-big1.txt")
    # lines = f.readlines()
    # for l in lines:
    #     m = Chem.MolFromSmiles(l.split(",")[0])
    #     print(m)
    # for i in range(1000):
    #     x = random.randint(0, len(lines) / 2)
    #     y = random.randint(len(lines) / 2, len(lines))
    #     source = lines[x].split(",")[0]
    #     dest = lines[y].split(",")[0]
    #     # source = str(data_train[x])
    #     # dest = str(data_train[y])
    #     results = interpolate_q(source, dest, steps, charset, model, latent_dim, width)
    #     # l = len(results)
    #     for result in results:
    #         # print(result[2])
    #         m = Chem.MolFromSmiles(result[2])
    #         if m != None:
    #             type += 1
    #             if str(result[2]) == str(source) or str(result[2]) == str(dest):
    #                 print('repeat!')
    #                 t += 1
    #             print(source)
    #             print(result[0], result[2])
    #             print(dest)
    # print(type, type / 10000, t)

if __name__ == '__main__':
    main()
