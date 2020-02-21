import sys
import numpy as np
from basic import *

# proc training data
train_feats_file = '../training_feats/face_d128_2M_trainingFeats.txt.npy'

train_feats = np.load(train_feats_file)

part_num = 10

record_num = len(train_feats)

part_len = int(record_num * 1.0 / part_num)

for pid in range(part_num - 1):
    train_sub_feats_file ='../training_feats/face_d128_2M_trainingFeats_part' + str(pid) + '.txt'
    np.save(train_sub_feats_file, train_feats[part_len * pid: part_len * (pid + 1)])


pid = part_num - 1
train_sub_feats_file ='../training_feats/face_d128_2M_trainingFeats_part' + str(pid) + '.txt'
np.save(train_sub_feats_file, train_feats[part_len * pid:])

# proc testing data
test_feats_file = '../testing_feats/face_d128_2M_testingFeats.txt.npy'

test_feats = np.load(test_feats_file)

part_num = 10

record_num = len(test_feats)

part_len = int(record_num * 1.0 / part_num)

for pid in range(part_num - 1):
    test_sub_feats_file ='../testing_feats/face_d128_2M_testingFeats_part' + str(pid) + '.txt'
    np.save(test_sub_feats_file, test_feats[part_len * pid: part_len * (pid + 1)])


pid = part_num - 1
test_sub_feats_file ='../testing_feats/face_d128_2M_testingFeats_part' + str(pid) + '.txt'
np.save(test_sub_feats_file, test_feats[part_len * pid:])


# proc validation data
valid_feats_file = '../validation_feats/face_d128_2M_validationFeats.txt.npy'

valid_feats = np.load(valid_feats_file)

part_num = 10

record_num = len(valid_feats)

part_len = int(record_num * 1.0 / part_num)

for pid in range(part_num - 1):
    valid_sub_feats_file ='../validation_feats/face_d128_2M_validationFeats_part' + str(pid) + '.txt'
    np.save(valid_sub_feats_file, valid_feats[part_len * pid: part_len * (pid + 1)])


pid = part_num - 1
valid_sub_feats_file ='../validation_feats/face_d128_2M_validationFeats_part' + str(pid) + '.txt'
np.save(valid_sub_feats_file, valid_feats[part_len * pid:])


