import numpy as np
from scipy.spatial import distance
import sys
import math
import pickle

#part_id = int(sys.argv[1])
leaf_id = int(sys.argv[1])

data_file = sys.argv[2] #'../real_data/leaf_data/face_d128_2M_originalData_LEAFID_' + str(leaf_id) + '.npy'
tau_file = sys.argv[3] #'../raw_labels/face_d128_2M_trainingFeats_smallSel_part' + str(part_id) + '_rawLabels.npy' 
query_file = sys.argv[4]
result_file_prefix = sys.argv[5]

data = np.load(data_file)
taus = np.load(tau_file)

# for train
#query_file = '../training_feats/face_d128_2M_trainingFeats_part' + str(part_id) + '.txt.npy'
queries = np.load(query_file)
predictions = []

for rid in range(queries.shape[0]):
    predict = []
    _query = queries[rid: rid + 1]
    res = distance.cdist(data, _query, 'cosine')
    tau_arr = taus[rid]
    for tau in tau_arr:
        res_num = (res <= tau).sum()
        predict.append(res_num)
        
    predictions.append(predict)

predictions = np.array(predictions)
#result_file = '../raw_labels_leaf/face_d128_2M_trainingFeats_smallSel_part' + str(part_id) + '_leaf_' + str(leaf_id) + '_rawLabels.npy'
result_file = result_file_prefix + str(leaf_id) + '_rawLabels.npy'
np.save(result_file, predictions)

