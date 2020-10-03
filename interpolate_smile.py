from __future__ import print_function
from sklearn.model_selection import train_test_split
import h5py
import numpy
import os
import pandas
import random
from functools import reduce
import argparse
import sample
from molecules.vaemodel import VAE
from molecules.util import decode_smiles_from_indexes
from molecules.utils import one_hot_array, one_hot_index
from rdkit import Chem

# source = 'Cc1ccnc(c1)NC(=O)Cc2cccc3c2cccc3'
# dest = 'COc1cc(Cc2cnc(N)nc2N)cc(OC)c1N(C)C'
latent_dim = 292
steps = 100
width = 120
length = 500000

def vector(smiles, charset):
    smiles_vector = []
    for c in smiles:
        charset_vector = [0] * len(charset)
        for index, value in enumerate(charset):
            if c == value:
                charset_vector[index] = 1
        smiles_vector.append(charset_vector)
    return smiles_vector
def interpolate(source, dest, steps, charset, model, latent_dim, width, results):

    # s1 = []
    # s2 = []
    # a = vector(source, charset)
    # b = vector(dest, charset)
    # print(a)
    # print(b)
    # s1.append(a)
    # s2.append(b)
    # s1 = numpy.array(s1)
    # s2 = numpy.array(s2)

    source_x_latent = model.encoder.predict(numpy.array(source)).argmax(axis=2)[0]

    dest_x_latent = model.encoder.predict(numpy.array(dest)).argmax(axis=2)[0]




    step = (dest_x_latent - source_x_latent)/float(steps)
    for i in range(steps):
        item = source_x_latent + (step * i)
        sampled = model.decoder.predict(item.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        results.append((i, item, sampled))
    return results

def main():
    results = []
    type = 0

    model = VAE()
    if os.path.isfile('vae_model7.h5'):
        model.load('vae_model7.h5', latent_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % 'vae_model7.h5')

    # train = []
    # test = []
    # train_idx, test_idx = map(numpy.array, train_test_split(smiles.index, test_size=0.20))
    # for smile in smiles[train_idx]:
    #     if type == 100:
    #         break
    #     else:
    #         train.append(vector(smile, charset))
    #     type += 1
    #
    # type = 0
    # for smile in smiles[test_idx]:
    #     if type == 100:
    #         break
    #     else:
    #         test.append(vector(smile, charset))
    #     type += 1
    #
    # for i in range(100):
    #     source = random.sample(train, 1)
    #     dest = random.sample(test, 1)
    #     print(source)
    #     print(dest)
    #     interpolate(source, dest, steps, charset, model, latent_dim, width, results)

    for i in range(10000):
        latent = [0] * latent_dim
        # s = []
        for j in range(len(latent)):
            latent[j] = random.uniform(-1, 1)
        # s = s.append(latent)
        latent = numpy.array(latent)
        sampled = model.decoder.predict(latent.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        m = Chem.MolFromSmiles(sampled)
        if m != None:
            type += 1
    print(type / 10000)


    # for result in results:
    #     m = Chem.MolFromSmiles(result[2])
    #     if m != None:
    #         type += 1
    #         print(result[0], result[2])
    #     else:
    #         print('失败')
    #         continue
    # print(type/100000)

if __name__ == '__main__':
    main()
