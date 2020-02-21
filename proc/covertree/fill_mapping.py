import numpy as np
import sys
from scipy.spatial import distance
import math

original_data_file = sys.argv[1]
leaf_centers_file = sys.argv[2]
query_file = sys.argv[3]
#x_dim = int(sys.argv[4])
mapping_file = sys.argv[4]
data_name = sys.argv[5]

data = np.load(query_file)
original_data = np.load(original_data_file)
leaf_centers = np.load(leaf_centers_file)
leaf_num = len(leaf_centers)
x_dim = data.shape[1] - 2

print(leaf_num)

if data_name == 'face' or data_name == 'fasttext_cos' or data_name == 'fasttext_eu':
    _dist = 'cosine'
else:
    _dist = 'euclidean'

sc_centers = np.zeros(leaf_num) #[0] * leaf_num
radius = np.zeros(leaf_num)  #[0] * leaf_num

for e in leaf_centers:
    lid = int(e[0])
    sc_centers[lid] = int(e[1])
    radius[lid] = e[2]

sc_centers = np.array(sc_centers, dtype='int')
print(sc_centers)

center_data = original_data[sc_centers]

mapping = np.zeros((data.shape[0], leaf_num), dtype=np.uint8)

for rid in range(data.shape[0]):
    record = data[rid]
    tau = record[x_dim]
    if rid == 0:
        res = distance.cdist(center_data, data[rid: rid+1, :x_dim], _dist)
        res = np.hstack(res)
    if rid > 0 and ( (record[:x_dim]==data[rid - 1, :x_dim]).all() == False ):
        res = distance.cdist(center_data, data[rid: rid+1, :x_dim], _dist)
        res = np.hstack(res)
    for lid in range(len(res)):
        c = res[lid]
        if (tau + radius[lid]) > c:
            mapping[rid, lid] = 1

np.save(mapping_file, mapping)

