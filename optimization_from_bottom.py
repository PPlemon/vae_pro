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
from scipy.spatial.distance import pdist

#gp = joblib.load('/data/tp/data/model/Gaussian_model_2000_5qed-sas.pkl')

from molecules.predicted_vae_model import VAE_prop
from rdkit import Chem
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model = VAE_prop()
modelname = '/data/tp/data/model/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
if os.path.isfile(modelname):
    model.load(35, 120, modelname, latent_rep_size=196)
else:
    raise ValueError("Model file %s doesn't exist" % modelname)


def decode_smiles_from_indexes(vec, charset):
    return "".join(map(lambda x: charset[x], vec)).strip()

def objective(x):
    # res = []
    # for i in x:
    #     res.append(i)
    return model.predictor.predict(x.reshape(1, 196))[0]*-1

def main():
    # latent = open('data/data_train(2000)_latent(5_qed-sas).pkl', 'rb')
    # latent = pickle.load(latent)
    # target = open('data/data_train(2000)_target(5_qed-sas).pkl', 'rb')
    # target = pickle.load(target)
    #data = open('data/bottom_mol(5_qed-sas).pkl', 'rb')
    #data = pickle.load(data)
    h5f = h5py.File('/data/tp/data/per_all_250000.h5', 'r')
    charset2 = h5f['charset'][:]
    data_train = h5f['smiles_train'][:]
    qed_train = h5f['qed_train'][:]
    sas_train = h5f['sas_train'][:]
    target_train = np.array(qed_train) * 5 - np.array(sas_train)
    charset1 = []
    for i in charset2:
        charset1.append(i.decode())
    #model = VAE_prop()
    #modelname = '/data/tp/data/model/predictor_vae_model_250000_0(5qed-sas)(std=1).h5'
    #if os.path.isfile(modelname):
    #    model.load(35, 120, modelname, latent_rep_size=196)
    #else:
    #    raise ValueError("Model file %s doesn't exist" % modelname)
    t = []
    for i in target_train:
        t.append(i)
    t = np.array(t)
    sort = t.argsort()
    bottom_2000 = sort[:2000]
    print(t[bottom_2000[:10]])
    res = []
    data = model.encoder.predict(data_train)

    for j in range(len(bottom_2000)):
        temp = []
        start = data[bottom_2000[j]]
        start_pro = t[bottom_2000[j]]
        result = minimize(objective, start, method='COBYLA')
        solution = result['x']
        # evaluation = objective(solution)
        old = model.decoder.predict(start.reshape(1, 196)).argmax(axis=2)[0]
        sampled = model.decoder.predict(solution.reshape(1, 196)).argmax(axis=2)[0]
        sampled = decode_smiles_from_indexes(sampled, charset1)
        m = Chem.MolFromSmiles(sampled)
        print(decode_smiles_from_indexes(old, charset1))
        print(sampled)
        if m != None:
            print('起点：', decode_smiles_from_indexes(old, charset1))
            print('起点属性值：', start_pro)
            print('终点：', sampled)
            #print('高斯预测属性值：', gp.predict(solution)[0])
            print('终点属性值：', model.predictor.predict(solution.reshape(1, 196))[0])
            print('有效\n')
            temp.append(start)
            temp.append(solution)
            temp.append(start_pro)
            temp.append(model.predictor.predict(solution.reshape(1, 196))[0])
            res.append(temp)
        # else:
        #     print('无效\n')

    print(res, len(res))
    optimization = open('data/optimization_result_from_bottom_CVAE(5qed-sas)(std=1).pkl', 'wb')
    pickle.dump(res, optimization)
    optimization.close()

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
