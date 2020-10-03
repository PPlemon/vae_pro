import h5py
import numpy
import os
import random
import base64
import pandas
from rdkit import Chem
from functools import reduce
import argparse
from molecules.vaemodel import VAE
from molecules.util import decode_smiles_from_indexes, vector
from molecules.utils import one_hot_array, one_hot_index

#source = 'C=CCc1ccc(OCC(=O)N(CC)CC)c(OC)c1'
#dest = 'C=C(C)CNc1ccc([C@H](C)C(=O)O)cc1'
latent_dim = 196
steps = 10
width = 120
length = 500000
SMILES_CHARS = [' ',
                '(', ')',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']
def interpolate(source, dest, steps, charset, model, latent_dim, width):
    s = []
    # source_just = source.ljust(width)
    # dest_just = dest.ljust(width)
    # print(source_just)
    # print(dest_just)
    # one_hot_encoded_fn = lambda row: map(lambda x: one_hot_array(x, len(charset)),
    #                                             one_hot_index(row, charset))
    # source_encoded = numpy.array(map(one_hot_encoded_fn, source_just))
    #
    # dest_encoded = numpy.array(map(one_hot_encoded_fn, dest_just))
    source_encoded = vector(source)
    dest_encoded = vector(dest)
    s.append(source_encoded)
    s.append(dest_encoded)
    s = numpy.array(s)
    latent = model.encoder.predict(s)
    source_x_latent = latent[0]
    dest_x_latent = latent[1]

    step = (dest_x_latent - source_x_latent)/float(steps)
    results = []
    for i in range(steps):
        sampled1 = ''
        item = source_x_latent + (step * i)
        sampled = model.decoder.predict(item.reshape(1, latent_dim)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset)
        sampled = bytes.decode(base64.b64decode(sampled.encode()))
        for c in range(len(sampled)):
            if sampled[c] in SMILES_CHARS:
                sampled1 += sampled[c]
        results.append((i, item, sampled1.rstrip()))

    return results

def main():
    type = 0
    t = 0
    # data = pandas.read_hdf('data/smiles_500k.h5', 'table')
    # keys = data['structure'].map(len) < 121
    # if length <= len(keys):
    #     data = data[keys].sample(n = length)
    # else:
    #     data = data[keys]
    # smiles = data['structure'].map(lambda x: list(x.ljust(120)))
    # charset = list(reduce(lambda x, y: set(y) | x, smiles, set()))
    charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S',
               'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
               'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6',
               '7', '8', '9', '+', '/']
    model = VAE()
    if os.path.isfile('vae_model9.h5'):
        model.load('vae_model9.h5', latent_size = latent_dim)
    else:
        raise ValueError("Model file %s doesn't exist" % 'vae_model9.h5')
    f = open(r"Smiles.txt")
    lines = f.readlines()


    for i in range(100):
        x = random.randint(0, len(lines)/2)
        y = random.randint(len(lines)/2, len(lines))
        source = lines[x].strip()
        print(source)
        dest = lines[y].strip()
        print(dest)
        results = interpolate(source, dest, steps, charset, model, latent_dim, width)
        for result in results:
            m = Chem.MolFromSmiles(result[2])
            if m != None:
                type += 1
                if str(result[2]) == str(source) or str(result[2]) == str(dest):
                    print('repeat!')
                    t += 1
            print(result[0], result[2])
        else:
            continue
    print(type/1000, t)

if __name__ == '__main__':
    main()
