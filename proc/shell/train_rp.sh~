#!/bin/bash

data=$1

data_file='../../data/'${data}'/real_data/'${data}'_originalData.npy'

# random partition
mkdir './'${data}'_randompartition/'
mkdir './'${data}'_randompartition/clusters/'
part_num=3
cluster_file_prefix='./'${data}'_randompartition/clusters/'${data}'_randompartition_cluster_'
python ../randompartition/randompartition.py ${data_file} ${part_num} ${cluster_file_prefix}


# generate training data for cover tree clusters
train_result_file='./raw_labels/'${data}'_trainingFeats_rawLabels.npy'
train_feats_file='./feats/'${data}'_trainingFeats.npy'
mkdir './raw_labels_leaf'
train_result_leaf_file='./raw_labels_leaf/'${data}'_trainingFeats_randompartition_cluster_'
python ../train/cal_leaf.py 0 ${cluster_file_prefix}'0.npy' ${train_result_file} ${train_feats_file} ${train_result_leaf_file}
python ../train/cal_leaf.py 1 ${cluster_file_prefix}'1.npy' ${train_result_file} ${train_feats_file} ${train_result_leaf_file}
python ../train/cal_leaf.py 2 ${cluster_file_prefix}'2.npy' ${train_result_file} ${train_feats_file} ${train_result_leaf_file}

# assemble training data
train_data_file='../../data/'${data}'/train/'${data}'_RandomPartition_trainingData.npy'
python ../train/proc_labels_leaf.py ${data_file} ${train_feats_file} ${train_result_file} 3 ${train_result_leaf_file} ${train_data_file} 


