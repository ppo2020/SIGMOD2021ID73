#!/bin/bash

data=$1

data_file='../../data/'${data}'/real_data/'${data}'_originalData.npy'
train_feats_file='./feats/'${data}'_trainingFeats.npy'
valid_feats_file='./feats/'${data}'_validationFeats.npy'
test_feats_file='./feats/'${data}'_testingFeats.npy'
mkdir './feats'

# splitting training, validation and testing queries
python ../train/selectTrainTestFeats.py ${data_file} ${train_feats_file} ${valid_feats_file} ${test_feats_file}

# generate labels
mkdir './raw_labels'
train_result_file='./raw_labels/'${data}'_trainingFeats_rawLabels.npy'
valid_result_file='./raw_labels/'${data}'_validationFeats_rawLabels.npy'
test_result_file='./raw_labels/'${data}'_testingFeats_rawLabels.npy'

python ../train/cal.py ${data_file} ${train_feats_file} ${train_result_file}
python ../train/cal.py ${data_file} ${valid_feats_file} ${valid_result_file}
python ../train/cal.py ${data_file} ${test_feats_file} ${test_result_file}

# assemble training, validation and testing data
mkdir '../../data/'${data}'/train/'
train_data_file='../../data/'${data}'/train/'${data}'_trainingData.npy'
valid_data_file='../../data/'${data}'/train/'${data}'_validationData.npy'
test_data_file='../../data/'${data}'/train/'${data}'_testingData.npy'

python ../train/proc_labels.py ${data_file} ${train_feats_file} ${train_result_file} ${train_data_file}
python ../train/proc_labels.py ${data_file} ${valid_feats_file} ${valid_result_file} ${valid_data_file}
python ../train/proc_labels.py ${data_file} ${test_feats_file} ${test_result_file} ${test_data_file}

