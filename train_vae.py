from __future__ import print_function

import argparse
import os
import h5py
import numpy as np
# import matplotlib.pyplot as plt


NUM_EPOCHS = 1000
BATCH_SIZE = 128
LATENT_DIM = 196
RANDOM_SEED = 1337


def main():
    # args = get_arguments()
    np.random.seed(RANDOM_SEED)

    from molecules.model import MoleculeVAE
    from molecules.utils import one_hot_array, one_hot_index, from_one_hot_array, \
        decode_smiles_from_indexes, load_dataset
    from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
    
    # data_train, data_test, charset = load_dataset('data/per_all_25000(index)h5')
    h5f = h5py.File('data/per_all_45(150).h5', 'r')
    data_train = h5f['smiles_train'][:]
    data_test = h5f['smiles_test'][:]
    charset = h5f['charset'][:]
    print(len(charset))
    print(charset)
    model = MoleculeVAE()
    if os.path.isfile('data/vae_model_45(150)(1).h5'):
        model.load(charset, 'data/vae_model_45(150)(1).h5', latent_rep_size=LATENT_DIM)
    else:
        model.create(charset, latent_rep_size=LATENT_DIM)

    check_pointer = ModelCheckpoint(filepath='data/vae_model_45(150)(1).h5', verbose=1, save_best_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.0001)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

    tbCallBack = TensorBoard(log_dir="TensorBoard/vae_model_45(150)(1)")

    history = model.autoencoder.fit(
        data_train,
        data_train,
        shuffle=True,
        epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[check_pointer, reduce_lr, early_stopping, tbCallBack],
        validation_data=(data_test, data_test)
    )
    # acc = history.history['acc']
    # val_acc = history.history['val_acc']
    #
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    #
    # epochs_range = history.epoch

    # plt.figure(figsize=(8, 8))
    #
    # plt.subplot(1, 2, 1)
    # plt.plot(epochs_range, acc, label='Training Accuracy')
    # plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    # plt.legend(loc='lower right')
    # plt.title('Training and Validation Accuracy')
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs_range, loss, label='Training Loss')
    # plt.plot(epochs_range, val_loss, label='Validation Loss')
    # plt.legend(loc='upper right')
    # plt.title('Training and Validation Loss')
    # plt.show()

if __name__ == '__main__':
    main()
