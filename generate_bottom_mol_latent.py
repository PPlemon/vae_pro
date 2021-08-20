import joblib
import pickle
import h5py
import numpy as np
import os
from molecules.predicted_vae_model import VAE_prop


model = VAE_prop()
# modelname = 'model/predictor_vae_model_w2v_30_new_250000_12260707(2).h5'
modelname = '/data/tp/data/model/predictor_vae_model_250000_12260707(5_qed-sas).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
data_train = h5f['smiles_train'][:50000]
qed_train = h5f['qed_train'][:50000]
sas_train = h5f['sas_train'][:50000]
target_train = np.array(qed_train) * 5 - np.array(sas_train)
t = []
for i in target_train:
    t.append(i)
t.sort(reverse=True)
bottom_2000 = t[-2000:]
res = []
for i in bottom_2000:
    temp = []
    idx = np.where(target_train == i)[0][0]
    temp.append(idx)
    temp.append(target_train[idx])
    temp.append(model.encoder.predict(data_train[idx].reshape(1, 120, 35)))
    res.append(temp)

bottom_mol = open('data/bottom_mol(5_qed-sas).pkl', 'wb')
pickle.dump(res, bottom_mol)
bottom_mol.close()
