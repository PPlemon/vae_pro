import h5py
h5f = h5py.File('data/per_all_base64_250000.h5', 'r')
smiles_train = h5f['smiles_train'][:]
smiles_test = h5f['smiles_test'][:]
sas_train = h5f['sas_train'][:]
sas_val = h5f['sas_val'][:]
sas_test = h5f['sas_test'][:]
# charset = h5f['charset'][:]
# print(charset)
print(sas_train)
print(sas_val)
print(len(smiles_train[0]), len(smiles_train[0][1]), len(smiles_train)+len(smiles_test))
print(smiles_test[0][119])
h5f.close()

