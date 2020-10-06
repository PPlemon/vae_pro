import os
import h5py
import base64
from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn.model_selection import train_test_split
from molecules.model import MoleculeVAE
from molecules.util import base32_vector, base64_vector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
base32_charset = [' ', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', '2', '3', '4', '5', '6', '7']
base64_charset = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                  'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                  'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', '+', '/']
base64_charset_120 = ['=', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                      'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n',
                      'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1', '2', '3', '4', '5', '6', '7',
                      '8', '9', '+', '/']
batch_size = 128
latent_dim = 196
epochs = 1000
RANDOM_SEED = 1337
def main():
    l = [41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    for i in l:
        np.random.seed(RANDOM_SEED)
        filename = 'data/per_all_base64_' + str(i) + '.h5'
        h5f = h5py.File(filename, 'r')
        data_train = h5f['smiles_train'][:]
        data_test = h5f['smiles_test'][:]
        model = MoleculeVAE()
        length = len(data_train[0])
        modelname = 'vae_model_base64_' + str(i) + '.h5'

        if os.path.isfile(modelname):
            model.load(base64_charset, modelname, latent_rep_size=latent_dim)
        else:
            model.create(base64_charset, max_length=length, latent_rep_size=latent_dim)
        check_pointer = ModelCheckpoint(filepath=modelname, verbose=1, save_best_only=True)

        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001)

        early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

        TensorBoardname = "TensorBoard/vae_model_base64_" + str(i)

        tbCallBack = TensorBoard(log_dir=TensorBoardname)

        print(data_train[0])
        history = model.autoencoder.fit(
            data_train,
            data_train,
            shuffle=True,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[check_pointer, reduce_lr, early_stopping, tbCallBack],
            validation_data=(data_test, data_test)
        )

if __name__ == '__main__':
    main()
