from cover_tree import *
import numpy as np
import sys
import os
from scipy.spatial.distance import euclidean, cosine


original_data_file = sys.argv[1] #'/import/sigmod05/1/scratch/yaoshuw/SphericalRangeCardEstimation/data/face/real_data/face_d128_2M_originalData.npy'

leaf_data_file_dir = sys.argv[2]
cover_tree_dir = sys.argv[3]
data_name = sys.argv[4]

if data_name == 'face' or data_name == 'fasttext_cos' or data_name == 'youtube':
    _base = 1.2
    distance = cosine
elif data_name == 'fasttext_eu':
    _base = 2
    distance = euclidean

real_data = np.load(original_data_file)

print('Shape : ', real_data.shape)

x_dim = real_data.shape[1]

leaf_data_file_prefix = os.path.join(leaf_data_file_dir, 'original_data')
if not os.path.exists(leaf_data_file_dir):
    os.makedirs(leaf_data_file_dir)
#cover_tree_dir = '../data/face_d128_2M/cover_tree/'
if not os.path.exists(cover_tree_dir):
    os.makedirs(cover_tree_dir)

cover_tree = CoverTree(real_data, distance, 
                leaf_data_file_prefix, cover_tree_dir,
                leafsize=200000, base=_base)

cover_tree._build()

