import numpy as np
import random
RANDOM_SEED = 1337
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
from molecules.model import MoleculeVAE
import os
import h5py
base64_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base64_charset_120 = ['=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', '+', '/']
base32_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7']
# 验证
h5f = h5py.File('data/per_all_base64_250000.h5', 'r')
data_train = h5f['smiles_train'][:]
data_val = h5f['smiles_val'][:]
data_test = h5f['smiles_test'][:]
print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
# charset = h5f['charset'][:]
length = len(data_train[0])
h5f.close()
model = MoleculeVAE()
if os.path.isfile('data/vae_model_base64_250000.h5'):
    model.load(base64_charset_120, length, 'data/vae_model_base64_250000.h5', latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % 'data/vae_model_base64_250000.h5')
data_test_vae = model.autoencoder.predict(data_test)
count0 = 0
count1 = 0
for m in range(len(data_test_vae)):
    item0 = data_test[m].argmax(axis=1)
    item1 = data_test_vae[m].argmax(axis=1)
    print(item0)
    # print(item1)
    # break
    for j in range(len(item0)):
        # 补空格时用于跳出循环
        # if item0[j] == 0:
          #   break
        count0 += 1
        if item0[j] != item1[j]:
            count1 += 1
print(count0, count1)
print((count0-count1)/count0)
