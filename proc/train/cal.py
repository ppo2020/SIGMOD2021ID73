import numpy as np
from scipy.spatial import distance
import sys
import math
import pickle

#part_id = int(sys.argv[1])

data_file = sys.argv[1]
query_file = sys.argv[2]
result_file = sys.argv[3]

#data_file = '../real_data/face_d128_2M_originalData.npy'
data = np.load(data_file)
data_num = data.shape[0]

# for train
#query_file = '../training_feats/face_d128_2M_trainingFeats_part' + str(part_id) + '.txt.npy'
queries = np.load(query_file)
predictions = []
selectivity = np.geomspace(0.0001, 1, 40)

for rid in range(queries.shape[0]):
    predict = []
    _query = queries[rid: rid + 1]
    res = distance.cdist(data, _query, 'cosine')
    res = sorted(np.hstack(res))
    # generate training data according to selectivity
    for sel in selectivity:
        _label = int(data_num * sel / 100)
        #assert (_label - 1) >= 0, "Labels should be >= 1"
        if (_label - 1) < 0:
            _label = 1
        predict.append(res[_label - 1])
        
    predictions.append(predict)

predictions = np.array(predictions)
#result_file = '../raw_labels/face_d128_2M_trainingFeats_smallSel_part' + str(part_id) + '_rawLabels.npy'
np.save(result_file, predictions)

