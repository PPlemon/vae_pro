import numpy as np
import random
RANDOM_SEED = 12260707
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
import h5py
import os
import joblib
import pickle
from sklearn.decomposition import PCA
from matplotlib import pyplot
from scipy.optimize import minimize


gp = joblib.load('model/Gaussian_model_2000.pkl')
from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem

h5f = h5py.File('data/per_all_250000.h5', 'r')
charset2 = h5f['charset'][:]
charset1 = []
for i in charset2:
    charset1.append(i.decode())
model = VAE_prop()
modelname = 'model/predictor_vae_model_250000_12260707.h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)

def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def objective(x):
    res = []
    for i in x:
        res.append(i)
    return gp.predict([res])[0]*-1

def latent_to_smiles(solution):
    from molecules.predicted_vae_model import VAE_prop
    from rdkit import Chem
    h5f = h5py.File('data/per_all_250000.h5', 'r')
    charset2 = h5f['charset'][:]
    charset1 = []
    for i in charset2:
        charset1.append(i.decode())
    model = VAE_prop()
    modelname = 'model/predictor_vae_model_250000_12260707.h5'
    if os.path.isfile(modelname):
        model.load(35, 120, modelname, latent_rep_size=196)
    else:
        raise ValueError("Model file %s doesn't exist" % modelname)
    sampled = model.decoder.predict(solution.reshape(1, 196)).argmax(axis=2)[0]
    sampled = decode_smiles_from_indexes(sampled, charset1)
    m = Chem.MolFromSmiles(sampled)
    if m != None:
        print('解码分子有效！')
        return sampled
    else:
        print('解码分子无效！')
        return sampled

def main():
    latent = open('data/data_train(2000)_latent.pkl', 'rb')
    latent = pickle.load(latent)
    target = open('data/data_train(2000)_target.pkl', 'rb')
    target = pickle.load(target)
    t = []
    for i in target:
        t.append(i)
    ind = t.index(max(t))
    ind1 = t.index(min(t))
    print('最大属性值：', t[ind])
    # print(latent[ind])
    print('最小属性值：', t[ind1])
    # print(latent[ind1])
    # pt = latent[0]
    # for i in range(554, 2000):
    pt = np.array(latent[2])
    print(pt)
    bounds = ()
    for i in range(196):
        t0 = min(latent[:][i])
        t1 = max(latent[:][i])
        # t0 = -0.1
        # t1 = 0.1
        # b = (-0.1, 0.1)
        bounds = bounds + ({'type': 'ineq', 'fun': lambda x: x[i] - t0},
                           {'type': 'ineq', 'fun': lambda x: -x[i] + t1})
    print(bounds)
    result = minimize(objective, pt, constraints=bounds, method='COBYLA')
    # print('Status : %s' % result['message'])
    # print('Total Evaluations: %d' % result['nfev'])
    solution = result['x']
    # evaluation = objective(solution)
    print('找到的最大属性值：', gp.predict([solution])[0])
    # print(gp.predict([latent[ind]])[0])
    print(solution)
    print(latent_to_smiles(solution))


    # solution = np.array(solution)
    # ind2 = np.where(latent == solution)
    #
    # print(ind2)
    # pca = PCA(n_components=2)
    # latent = pca.fit_transform(latent)
    # print(type(latent))
    # x = latent[:, 0]
    # y = latent[:, 1]
    # print(x)
    # target = objective(qed_test, sas_test)
    # print(target)
    # figure = pyplot.figure()
    # axis = figure.gca(projection='3d')
    # axis.plot_trisurf(x, y, target, cmap='YlGnBu')
    # pyplot.show()



if __name__ == '__main__':
    main()
