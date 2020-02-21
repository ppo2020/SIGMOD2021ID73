import numpy as np
import sys
import os

leaf_file_dir = sys.argv[1]
greedy_cluster_file = sys.argv[2]
data_name = sys.argv[3]

leaf_num = len(os.listdir(leaf_file_dir))

if data_name == 'face':
    leaf_max_size = 700000
elif data_name == 'fasttext_cos' or data_name == 'fasttext_eu':
    leaf_max_size = 400000
else:
    leaf_max_size = 130000

#leaf_file_prefix = '../leaf_original_dir/original_data_LEAF_ID_' #0.npy'
leaf_file_prefix = os.path.join(leaf_file_dir, 'original_data_LEAF_ID_')

leaf_lens = []
for lid in range(leaf_num):
    leaf_file = leaf_file_prefix + str(lid) + '.npy'
    if os.path.isfile(leaf_file):
        leaf_data = np.load(leaf_file)
        leaf_lens.append([lid, leaf_data.shape[0] ])

# greedy cluster, such that each leaf number is no larger than 'leaf_max_size'
# and with the minimal leaf_num

leaf_lens = sorted(leaf_lens, key = lambda x : x[1])
leaf_sum = np.sum([i[1] for i in leaf_lens])
print('Leaf Sum : ', leaf_sum)




leaf_clusters = []
script = 0
leaf_c = []
leaf_s = 0
i = 0
for i in range(len(leaf_lens)):
    leaf_len = leaf_lens[i]
    if (leaf_len[1] + leaf_s) > leaf_max_size:
        leaf_clusters.append(leaf_c)
        leaf_c = [leaf_len]
        leaf_s = leaf_len[1]
    else:
        leaf_c.append(leaf_len)
        leaf_s += leaf_len[1]
    # if the last one
    if i == len(leaf_lens) - 1:
        leaf_clusters.append(leaf_c)

print(len(leaf_clusters))

import pickle

#with open('face_d128_2M_covertree_greedy_cluster_leaf_IDS', 'wb') as f:
with open(greedy_cluster_file, 'wb') as f:
    pickle.dump(leaf_clusters, f)




