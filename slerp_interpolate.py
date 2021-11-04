import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem
from rdkit.Chem import Draw
from molecules.util import vector120


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()


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
    #source = 'CCCC(=O)Nc1ccc(OC[C@H](O)CNC(C)C)c(C(C)=O)c1'
    source = 'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1'
    dest = 'CCCNC[C@H](O)COc1ccccc1C(=O)CCc1ccccc1'


    h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
    # data_train = h5f['smiles_train'][:]
    # data_val = h5f['smiles_val'][:]
    data_test = h5f['smiles_test'][:5000]
    logp_test = h5f['logp_test'][:5000]
    # print(len(data_train), len(data_val), len(data_test), len(data_train[0]))
    charset2 = h5f['charset'][:]
    charset1 = []
    for i in charset2:
        charset1.append(i.decode())

    length = len(data_test[0])
    charset = len(data_test[0][0])
    h5f.close()
    steps = 100
    latent_dim = 196
    width = 120
    model = VAE_prop()
    result = []
    modelname = '/data/tp/data/model/CVAE/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'

    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)

    source_encoded = vector120(source, charset1)
    dest_encoded = vector120(dest, charset1)

    source_encoded = np.array(source_encoded)
    dest_encoded = np.array(dest_encoded)

    source_x_latent = model.encoder.predict(source_encoded.reshape(1, width, len(charset1)))
    dest_x_latent = model.encoder.predict(dest_encoded.reshape(1, width, len(charset1)))
    source_x_latent = source_x_latent[0]
    dest_x_latent = dest_x_latent[0]

    for i in range(steps):
        # item = source_x_latent + (step * i)
        item = slerp(i/steps, source_x_latent, dest_x_latent)
        sampled = model.decoder.predict(item.reshape(1, latent_dim)).argmax(axis=2)[0]
        s = decode_smiles_from_indexes(sampled, charset1)
        m = Chem.MolFromSmiles(s)
        if m != None:
            if s not in result:
                result.append(s)
            print(s)
            #print(model.predictor.predict(item.reshape(1, 196))[0])
            # f = 'data/picture/' + str(i) + '.png'
            # Draw.MolToFile(m, f, size=(150, 100))
    #result1 = list(set(result))
    print(len(result))
    print(result)
if __name__ == '__main__':
    main()
