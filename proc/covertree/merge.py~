import numpy as np
import sys
import pickle
import copy
import os

leaf_file_dir = sys.argv[1]
data_file = sys.argv[2]
greedy_cluster_file = sys.argv[3]
output_file_prefix = sys.argv[4]

#leaf_num = 368
leaf_num = len(os.listdir(leaf_file_dir))

leaf_IDs_arr = []
#leaf_file_prefix = '../leaf_original_dir/original_data_LEAF_ID_'
leaf_file_prefix = os.path.join(leaf_file_dir, 'original_data_LEAF_ID_')

data = np.load(data_file)

for lid in range(leaf_num):
    leaf_file = leaf_file_prefix + str(lid) + '.npy'
    if os.path.isfile(leaf_file):
        leaf_IDs = np.load(leaf_file)
        leaf_IDs_arr.append(leaf_IDs)

#with open('./face_d128_2M_covertree_greedy_cluster_leaf_IDS', 'rb') as f:
with open(greedy_cluster_file, 'rb') as f:
    greedy_clusters = pickle.load(f)

new_leaf_num = len(greedy_clusters)

count = 0

for lid in range(new_leaf_num):
    cluster = greedy_clusters[lid]
    old_lid = int(cluster[0][0])
    _ids_ = list( copy.deepcopy(leaf_IDs_arr[old_lid]) )
    for cid in range(1, len(cluster), 1):
        old_lid = int(cluster[cid][0])
        _ids_ += list( leaf_IDs_arr[old_lid] )
    leaf_data = data[_ids_]
    leaf_data = np.array(leaf_data)
    count += leaf_data.shape[0]
    print('Shape', leaf_data.shape)
    #np.save('./face_d128_2M_originalData_LEAFID_' + str(lid) + '.npy', leaf_data)
    np.save(output_file_prefix + str(lid) + '.npy', leaf_data)

print(data.shape[0], count)
