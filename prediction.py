import keras
from keras.models import Sequential
from sklearn.decomposition import PCA
from keras.layers import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from functools import reduce
from keras import optimizers
# import matplotlib.pyplot as plt
import h5py
import os
import pickle
import pandas
import zlib
import base64
import lzma
# import lda
import struct
import random
import numpy

latent_rep_size = 196
batch_size = 64
epochs = 1000
max_length = 120
n_topics = 80
steps = 50

# 获取ae预处理数据
def load_dataset(filename, split = True):
    print(filename)
    print(os.path.isfile(filename))
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
        property_train = h5f['property_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    property_test = h5f['property_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, property_train, property_test, charset)
    else:
        return (data_test, property_test, charset)

# 获取ae编码后数据
def load_property(filename, split = True):
    h5f = h5py.File(filename, 'r')
    data = h5f['latent_vectors'][:]
    property = h5f['property'][:]
    data_train, data_test, property_train, property_test = train_test_split(data, property, test_size=0.20)
    # if split:
    #     data_train = data[train_idx]
    #     property_train = property[train_idx]
    # else:
    #     data_train = None
    # data_test = data[test_idx]
    # property_test = property[test_idx]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return (data_train, data_test, property_train, property_test, charset)
    else:
        return (data_test, property_test, charset)
# 插值
def interpolate(source, dest, steps):
    step = (dest - source)/float(steps)
    results = []
    for i in range(steps):
        item = source + (step * i)
        sampled = basedecoder(item)
        results.append( (i, item, sampled) )
    return results

#zlib压缩
def zl(smile):
    num = []
    smiles = smile.encode()
    compressed = zlib.compress(smiles)
    # while compressed:
    #     s = struct.unpack('B', compressed[-1:])[0]/250
    #     compressed = compressed[:-1]
    #     num.append(s)
    for c in compressed:
        c = round(c / 250, 1)
        if c > 1:
            c = round(random.random(), 1)
        num.append(c)
    return num
# 不压缩
def zs(smile):
    num = []
    smiles = smile.encode().ljust(100)
    # compressed = zlib.compress(smiles).ljust(72)
    # while smiles:
    #     s = struct.unpack('B', smiles[-1:])[0]/100
    #     smiles = smiles[:-1]
    #     num.append(s)
    for c in smiles:
        c = round(c/100, 1)
        if c > 1:
            c = round(random.random(), 1)
        num.append(c)
    return num
def vector(smile, charset):
    vec = [0] * len(charset)
    smile = smile.ljust(120)
    for c in smile:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec
def zlibcharset(smiles):
    compresseds = []
    for smile in smiles:
        smile = smile.ljust(120)
        compressed = zlib.compress(smile.encode())
        compresseds.append(compressed)
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset
def zlibvector(smile, charset):
    smile = smile.ljust(120)
    compressed = zlib.compress(smile.encode())
    vec = [0] * len(charset)
    num = []
    for c in compressed:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec
def basecharset(smiles):
    compresseds = []
    for smile in smiles:
        smile = smile.ljust(120)
        compressed = base64.b64encode(smile.encode())
        compresseds.append(compressed)
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset

def basevector(smile, charset):
    smile = smile.ljust(120)
    compressed = base64.b64encode(smile.encode())
    vec = [0] * len(charset)
    num = []
    for c in compressed:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec
def lzmacharset(smiles):
    compresseds = []
    for smile in smiles:
        smile = smile.ljust(120)
        compressed = lzma.compress(smile.encode())
        compresseds.append(compressed)
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset
def lzmavector(smile, charset):
    smile = smile.ljust(120)
    compressed = lzma.compress(smile.encode())
    vec = [0] * len(charset)
    num = []
    for c in compressed:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec
def lzw(string):
    dictionary = {chr(i): i for i in range(1, 128)}
    last = 129
    p = ""
    result = []
    for c in string:
        pc = p + c
        if pc in dictionary:
            p = pc
        else:
            result.append(dictionary[p])
            dictionary[pc] = last
            last += 1
            p = c
    if p != '':
        result.append(dictionary[p])
    return result
def lzwcharset(smiles):
    compresseds = []
    for smile in smiles:
        compresseds.append(lzw(smile))
    charset = list(reduce(lambda x, y: set(y) | x, compresseds, set()))
    return charset
def lzwvector(smile,charset):
    compressed = lzw(smile)
    vec = [0] * len(charset)
    for c in compressed:
        for index, value in enumerate(charset):
            if c == value:
                vec[index] += 1
    return vec
# 获取smiles和属性值

h5f = h5py.File('data/per_all_latent_44(120).h5', 'r')
smiles_train_latent = h5f['smiles_train_latent'][:]
smiles_test_latent = h5f['smiles_test_latent'][:]
logp_train = h5f['logp_train'][:]
logp_test = h5f['logp_test'][:]
# qed = open('qed.pkl', 'rb')
# qed = pickle.load(qed)
# qed_train = qed[:225000]
# qed_test = qed[225000:]
qed_train = h5f['qed_train'][:]
qed_test = h5f['qed_test'][:]
sas_train = h5f['sas_train'][:]
sas_test = h5f['sas_test'][:]
# charset = h5f['charset'][:]
print(smiles_train_latent[0])
print(logp_train[0])
# 获取ae预处理数据
# data_train, data_test, property_train, property_test, charset = load_dataset('data/processed-big1.h5')
# 获取ae编码后数据
# data_train, data_test, property_train, property_test, charset = load_property('data/encoded.h5')

# 主成分分析降维
# pca = PCA(n_components=80)
# data_train = pca.fit_transform(data_train)
# data_test = pca.fit_transform(data_test)

# lda降维
# ldamodeltrain = lda.LDA(n_topics=n_topics, n_iter=100, random_state=1) #初始化模型, n_iter迭代次数
# ldamodeltrain.fit(data_train)
# data_train = np.array(ldamodeltrain.doc_topic_[:])
# ldamodeltest = lda.LDA(n_topics=n_topics, n_iter=100, random_state=1) #初始化模型, n_iter迭代次数
# ldamodeltest.fit(data_test)
# data_test = np.array(ldamodeltest.doc_topic_[:])

# 定义输入维度
input_shape = (latent_rep_size,)

# 回调函数
checkpointer = ModelCheckpoint(filepath = 'per_logp_model_44(120)(1).h5',
                                   verbose = 1,
                                   save_best_only = True)

reduce_lr = ReduceLROnPlateau(monitor = 'val_loss',
                                  factor = 0.2,
                                  patience = 3,
                                  min_lr = 0.0001)

early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=2)

# 模型
model = Sequential()
# model.add(Flatten(name='flatten_1'))
model.add(Dense(2000, activation='relu', input_shape=input_shape))
model.add(Dense(2000, activation='relu'))
model.add(Dense(1, activation='linear'))


# build模型
model.build((None, 196))
model.summary()

# 编译模型
adam = optimizers.Adam(lr=0.001)
model.compile(loss='mae', optimizer='Adam', metrics=['accuracy'])

# 训练模型
history = model.fit(smiles_train_latent, logp_train,
                    shuffle=True,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    callbacks = [checkpointer, reduce_lr, early_stopping],
                    validation_data=(smiles_test_latent, logp_test))

# 验证模型
# score = model.evaluate(smiles_test_latent, qed_test, verbose=0)
# print('Test loss', score[0])
# print('Test accuracy', score[1])

# 预测
# Y_predict = model.predict(smiles_train_latent)
# print('property', Y_predict[0])

# acc = history.history['acc']
# val_acc = history.history['val_acc']
#
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = history.epoch
#
# plt.figure(figsize=(8, 8))

# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title('Training and Validation Accuracy')

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title('Training and Validation Loss')
# plt.show()
