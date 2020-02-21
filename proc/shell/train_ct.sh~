#!/bin/bash

data=$1

data_file='../../data/'${data}'/real_data/'${data}'_originalData.npy'

# build cover tree with few length
mkdir './'${data}'_covertree/'
mkdir './'${data}'_covertree/leaf_data/'
mkdir './'${data}'_covertree/model/'
leaf_data_dir='./'${data}'_covertree/leaf_data/'
cover_tree_dir='./'${data}'_covertree/model/'
rm -r './'${data}'_covertree/leaf_data/*'
rm -r './'${data}'_covertree/model/*'
python ../covertree/buildCT.py ${data_file} ${leaf_data_dir} ${cover_tree_dir} ${data}

# collect greedy cluster info
greedy_cluster_file='../../data/'${data}'/train/'${data}'_covertree_greedy_cluster_leaf_IDS'
python ../covertree/greedy_cluster.py ${leaf_data_dir} ${greedy_cluster_file} ${data}
# merge clusters
mkdir './'${data}'_covertree/clusters'
cluster_file_prefix='./'${data}'_covertree/clusters/'${data}'_covertree_cluster_'
python ../covertree/merge.py ${leaf_data_dir} ${data_file} ${greedy_cluster_file} ${cluster_file_prefix} 

# generate training data for cover tree clusters
train_result_file='./raw_labels/'${data}'_trainingFeats_rawLabels.npy'
train_feats_file='./feats/'${data}'_trainingFeats.npy'
mkdir './raw_labels_leaf'
train_result_leaf_file='./raw_labels_leaf/'${data}'_trainingFeats_covertree_cluster_'
python ../train/cal_leaf.py 0 ${cluster_file_prefix}'0.npy' ${train_result_file} ${train_feats_file} ${train_result_leaf_file} 
python ../train/cal_leaf.py 1 ${cluster_file_prefix}'1.npy' ${train_result_file} ${train_feats_file} ${train_result_leaf_file}
python ../train/cal_leaf.py 2 ${cluster_file_prefix}'2.npy' ${train_result_file} ${train_feats_file} ${train_result_leaf_file} 

# fill into mapping file 
leaf_centers_file='./'${data}'_covertree/leaf_data/original_data_CENTERS.npy'
#train_feats_file='./feats/'${data}'_trainingFeats.npy'
#valid_feats_file='./feats/'${data}'_validationFeats.npy'
#test_feats_file='./feats/'${data}'_testingFeats.npy'
train_data_file='../../data/'${data}'/train/'${data}'_trainingData.npy'
valid_data_file='../../data/'${data}'/train/'${data}'_validationData.npy'
test_data_file='../../data/'${data}'/train/'${data}'_testingData.npy'
mapping_file_train='../../data/'${data}'/train/'${data}'_covertree_mapping_train.npy'
mapping_file_valid='../../data/'${data}'/train/'${data}'_covertree_mapping_valid.npy'
mapping_file_test='../../data/'${data}'/train/'${data}'_covertree_mapping_test.npy'
python ../covertree/fill_mapping.py ${data_file} ${leaf_centers_file} ${train_data_file} ${mapping_file_train} ${data}
python ../covertree/fill_mapping.py ${data_file} ${leaf_centers_file} ${valid_data_file} ${mapping_file_valid} ${data}
python ../covertree/fill_mapping.py ${data_file} ${leaf_centers_file} ${test_data_file} ${mapping_file_test} ${data}

# assemble training data
train_data_file='../../data/'${data}'/train/'${data}'_CoverTree_trainingData.npy'
python ../train/proc_labels_leaf.py ${data_file} ${train_feats_file} ${train_result_file} 3 ${train_result_leaf_file} ${train_data_file} 


