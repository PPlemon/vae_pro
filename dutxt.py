# import pandas

# def isfloat(s):import smiles
#     s = str(s)
#     if s.count('.') ==1:
#         left = s.split('.')[0]
#         right = s.split('.')[1]
#         if right.isdigit():
#             if left.count('-')==1 and left.startswith('-'):
#                 num = left.split['-'][-1]
#                 if num.isdigit():
#                     return True
#             elif left.isdigit():
#                 return True
#     return False

# f = open(r"attribute.txt")
# lines = f.readlines()
# t = 0
# s = ''
# a = ''
# for line in lines:
#     if t == 0:
#         for c in line:
#             if c == '\n':
#                 break
#             s += c
#         s = s + ','
#         t = 1
#         continue
#     if t == 1:
#         for c in line:
#             if c == '\n':
#                 break
#             a += c
#         if not isfloat(a):
#             print(a)
#             s = ''
#             t = 0
#             a = ''
#             continue
#         s += a
#         filename = 'data/smiles-end.txt'
#         with open(filename, 'a') as f:
#             f.write(s + "\n")
#         s = ''
#         t = 0
#         a = ''
#         continue

# f = open(r"data/substances.txt")
# lines = f.readlines()
# for line in lines:
#     s = ''
#     for c in line:
#         if c == ' ':
#             break
#         s += c
#     filename = 'Smiles.txt'
#     with open(filename, 'a') as f:
#         f.write(s + "\n")

# df = pandas.read_csv('data/zinc.csv', header = 0, index_col=0)
# df = df.rename(columns={'smiles':'smiles', 'logP':'logp', 'qed':'qed', 'SAS':'sas'})
# df.to_hdf('data/zinc-1.h5', 'table', format = 'table', data_columns = True)
import base64

# m = 0
# f = open(r"Smiles_25000_test.txt")
# lines = f.readlines()
# for i in lines:
#     m = max(m, len(i))
# print(m)

# compressed = base64.b32encode(lines[0].encode())
# compressed = compressed.ljust(192)
# print(compressed)

# for line in lines:
#     print(line)
#     break
#     s = ''
#     for c in line:
#         if c == ' ':
#             break
#         s += c
#     filename = 'Smiles.txt'
#     with open(filename, 'a') as f:
#         f.write(s + "\n")

# GPU测试
# import tensorflow as tf
# import os
#
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# a = tf.constant(1.)
# b = tf.constant(2.)
# print(a+b)
#
# print('GPU:', tf.test.is_gpu_available())

# import h5py
# h5f = h5py.File('data/per_all_base64_test.h5', 'r')
# # smiles_train = h5f['smiles_train'][:]
# smiles_test = h5f['smiles_test'][:]
# # charset = h5f['charset'][:]
# s = smiles_test[0][0]
# # print(charset)
# print(s)