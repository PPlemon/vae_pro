import numpy as np
import random
# RANDOM_SEED = 12260707
# random.seed(RANDOM_SEED)
# np.random.seed(RANDOM_SEED)
import h5py
import os
import pickle
from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem
from rdkit.Chem import Draw
from molecules.util import vector120, get_w2v_vector
from keras.models import Model, load_model



def slerp(val, low, high):
    if val <= 0:
        return low
    elif val >= 1:
        return high
    elif np.allclose(low, high):
        return low
    # 求反余弦arccos   np.dot求矩阵乘法得余弦值   np.linalg.norm求范数
    # 得到向量夹角
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    # 求正弦
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def main():
    source = 'CCCC(=O)Nc1ccc(OC[C@H](O)CNC(C)C)c(C(C)=O)c1'
    dest = 'CCCNC[C@H](O)COc1ccccc1C(=O)CCc1ccccc1'


    h5f = h5py.File('data/per_all_w2v_30_new_250000.h5', 'r')
    # data_train = h5f['smiles_train'][:]
    # data_val = h5f['smiles_val'][:]
    data_test = h5f['smiles_test'][:5000]
    logp_test = h5f['logp_test'][:5000]
    # print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
    w2v_vector = open('data/w2v_vector_30_new.pkl', 'rb')
    w2v_vector = pickle.load(w2v_vector)

    length = len(data_test[0])
    charset = len(data_test[0][0])
    h5f.close()

    steps = 100
    latent_dim = 196
    width = 120
    model = VAE_prop()
    result = []
    modelname = 'model/predictor_vae_model_w2v_30_new_250000_12260707(2).h5'

    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)

    w2v = load_model('model/word2vec.h5')
    embeddings = w2v.get_weights()[0]
    normalized_embeddings = embeddings / (embeddings ** 2).sum(axis=1).reshape((-1, 1)) ** 0.5

    word2id = open('data/word2id.pkl', 'rb')
    id2word = open('data/id2word.pkl', 'rb')

    word2id = pickle.load(word2id)
    id2word = pickle.load(id2word)

    def most_similar(w):
        sims = np.dot(normalized_embeddings, w)
        sort = sims.argsort()[::-1]
        sort = sort[sort > 0]
        return [(id2word[i], sims[i]) for i in sort[:1]]
        # return v

    source_encoded = get_w2v_vector(source, w2v_vector)
    dest_encoded = get_w2v_vector(dest, w2v_vector)
    source_encoded = np.array(source_encoded)
    dest_encoded = np.array(dest_encoded)
    source_x_latent = model.encoder.predict(source_encoded.reshape(1, width, charset))
    dest_x_latent = model.encoder.predict(dest_encoded.reshape(1, width, charset))
    source_x_latent = source_x_latent[0]
    dest_x_latent = dest_x_latent[0]

    for i in range(steps):
        item = slerp(i/steps, source_x_latent, dest_x_latent)
        sampled = model.decoder.predict(item.reshape(1, latent_dim))[0]
        s = ''
        for j in range(len(sampled)):
            s += most_similar(sampled[j])[0][0]
        s = s.strip()
        m = Chem.MolFromSmiles(s)
        if m != None:
            result.append(s)
            print(s)
    result1 = list(set(result))
    print(len(result1))
    print(result1)
if __name__ == '__main__':
    main()
