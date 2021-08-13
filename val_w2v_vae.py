import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
from molecules.predicted_vae_model import VAE_prop
import os
import h5py
import pickle
import base64
from keras.models import Model,load_model
from rdkit import Chem

# 验证
h5f = h5py.File('/data/tp/data/per_all_w2v_30_new_250000.h5', 'r')
#data_train = h5f['smiles_train'][:]
#data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
#print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
# charset = h5f['charset'][:]
length = len(data_test[0])
charset = len(data_test[0][0])
h5f.close()
model = VAE_prop()
if os.path.isfile('/data/tp/data/model/predictor_vae_model_w2v_30_new_250000_12260707(2).h5'):
    model.load(charset, length, '/data/tp/data/model/predictor_vae_model_w2v_30_new_250000_12260707(2).h5', latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % '/data/tp/data/model/predictor_vae_model_w2v_30_new_250000_12260707(2).h5')
data_test_vae = model.vae_predictor.predict(data_test)[0]

model = load_model('model/word2vec.h5')
embeddings = model.get_weights()[0]
normalized_embeddings = embeddings / (embeddings**2).sum(axis=1).reshape((-1, 1))**0.5


word2id = open('data/word2id.pkl', 'rb')
id2word = open('data/id2word.pkl', 'rb')

word2id = pickle.load(word2id)
id2word = pickle.load(id2word)


def most_similar(w):
    sims = np.dot(normalized_embeddings, w)
    sort = sims.argsort()[::-1]
    sort = sort[sort > 0]
    return [(id2word[i],sims[i]) for i in sort[:1]]
    #return v

t = 0
tt = 0
ttt = 0
#count  = 0
#count1 = 0
for i in range(5000):
    #item0 = data_test[m].argmax(axis=1)
    #item1 = data_test_vae[m].argmax(axis=1)
    #print(item0)
    # print(item1)
    # break
    s0 = ''
    item0 = data_test[i]
    for n in range(len(item0)):
        s0 += most_similar(item0[n])[0][0]     
    s0 = s0.strip()
    for m in range(1):
        item = data_test_vae[i]
#    print(item)
        s = ''
        for j in range(len(item)):
#        print(most_similar(item[j])[0])
            s+=most_similar(item[j])[0][0]
        s = s.strip()
        m = Chem.MolFromSmiles(s)
        if m!= None:
            t+=1
            if s != s0:
                print(s0)
                print(s)
                ttt += 1
#    print(s)
print(t)
print(t/5000)
print(tt)
print(ttt)