from gensim.models import word2vec
import h5py
import jieba
import pandas as pd
import pickle
from keras.models import Model,load_model
#data = pd.read_hdf('/data/tp/data/zinc-1.h5', 'table')
#smiles = data['smiles']
#logp = data['logp']
#qed = data['qed']
#sas = data['sas']
#sentences = []
#with open('data.txt', 'w') as f:
#    for s in smiles:
#        s = s.ljust(120)
#        split_words = [x  for x in s]
#        result = ' '.join(split_words)
#        f.writelines("{}\n".format(result))


#with open('data.txt', 'r') as f1:
#    for line in f1.readlines():
#        c = line
#        print(c, len(c), type(c))
#        break


#result = word2vec.LineSentence('data.txt')
#sentences = []
#for smiles in result:
#    smiles = smiles + [' ']*(120-len(smiles))
#    sentences.append(smiles)
#model = word2vec.Word2Vec(sentences, vector_size = 120, min_count=0, window=50)
#model.save('w2v_120.model')

#model = word2vec.Word2Vec.load('w2v_120.model')
model = load_model('./word2vec.h5')
filename = '/data/tp/data/per_all_250000.h5'
h5f = h5py.File(filename, 'r')
charset = h5f['charset'][:]
print(charset)
w2v_vector = {}
for s in charset:
    s = s.decode()
    print(s)
    w2v_vector[s] = model.wv[s]
print(w2v_vector['C'])
#output_smiles = open('w2v_vector_120.pkl', 'wb')
#pickle.dump(w2v_vector, output_smiles)
#output_smiles.close()


#def cut_word():
