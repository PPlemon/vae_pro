import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import joblib
import pickle
from molecules.predicted_vae_model import VAE_prop
from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def score_mae(output, target):
    return sum(abs(output-target))/len(output)


def main():

    h5f = h5py.File('data/per_all_250000.h5', 'r')
    # data_train = h5f['smiles_train'][:]
    # data_val = h5f['smiles_val'][:]
    data_train = h5f['smiles_train'][:5000]
    logp_train = h5f['logp_train'][:5000]
    qed_train = h5f['qed_train'][:5000]
    sas_train = h5f['sas_train'][:5000]
    target_train = np.array(qed_train)*5 - np.array(sas_train)
    print(qed_train[0], sas_train[0], target_train[0])

    length = len(data_train[0])
    charset = len(data_train[0][0])
    h5f.close()

    kernel = C(0.1, (0.001, 0.1)) * RBF(0.5, (1e-4, 10))

    model = VAE_prop()

    modelname = 'model/predictor_vae_model_250000_12260707.h5'

    if os.path.isfile(modelname):
        model.load(charset, length, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)
    data_train_vae = model.encoder.predict(data_train)

    latent = open('data/data_train(5000)_latent.pkl', 'wb')
    pickle.dump(data_train_vae, latent)
    latent.close()

    target = open('data/data_train(5000)_target.pkl', 'wb')
    pickle.dump(target_train, target)
    target.close()


    model_gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=100, alpha=0.01)

    model_gp.fit(data_train_vae, target_train)

    joblib.dump(model_gp, 'model/Gaussian_model_5000.pkl')

    y_pred = model_gp.predict(data_train_vae)

    score = score_mae(y_pred, target_train)

    print(score)

if __name__ == '__main__':
    main()
