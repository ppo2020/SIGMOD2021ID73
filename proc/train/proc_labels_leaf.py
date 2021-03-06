import sys
import numpy as np

#cid = int(sys.argv[1])

real_data_file = sys.argv[1]
data_file = sys.argv[2]
label_file = sys.argv[3]
leaf_num = int(sys.argv[4])
label_leaf_file_prefix = sys.argv[5]
output_file = sys.argv[6]

real_data = np.load(real_data_file)
data_num = real_data.shape[0]
#data_num = 2000000
#leaf_num = 3

# read original data
#data_file = '../../training_feats/glove_50_trainingFeats_part' + str(cid) + '_d64_binary_codes.npy'
#data_file = '../../training_feats/face_d128_2M_trainingFeats_part' + str(cid) + '.txt.npy'
data = np.load(data_file)

x_dim = data.shape[1]

print("Shape : ", data.shape)

#../../src/mixlabels_leaf/youtube_norm_trainingFeats_part0_LEAFID0.npy
#label_file = '../../raw_labels/face_d128_2M_trainingFeats_smallSel_part' + str(cid) + '_rawLabels.npy'
labels = np.load(label_file)

#print(labels[0], labels[1], len(labels))

selectivity = np.geomspace(0.0001, 1, 40)

tau_max_per_record = len(selectivity)

# save mixlabels file
data_mixlabels = np.zeros((data.shape[0] * tau_max_per_record, data.shape[1] + 1 + 1 + leaf_num))


# read leaf label files
label_leafs = []
for i in range(leaf_num):
    #_label_leaf = np.load('../../raw_labels_leaf/face_d128_2M_trainingFeats_smallSel_part' + str(cid) + '_leaf_' + str(i) + '_rawLabels.npy')
    _label_leaf = np.load(label_leaf_file_prefix + str(i) + '_rawLabels.npy')
    label_leafs.append(_label_leaf)

sc_f = []

selected_tau = 3

delta_leaf = 0
for rid in range(data.shape[0]):
    r_ = data[rid]
    np.random.seed(rid)
    sc_ = np.random.choice(tau_max_per_record, selected_tau, replace=False)
    t_max_script = np.max(sc_)
    for i in range(t_max_script):
        tau = labels[rid, i]
        _label = int(data_num * selectivity[i] / 100)
        if (_label - 1) < 0:
            _label = 1
        
        data_mixlabels[rid * tau_max_per_record + i, :x_dim] = data[rid]
        data_mixlabels[rid * tau_max_per_record + i, x_dim] = tau
        # insert leaf
        leaf_cum = 0
        for _l in range(leaf_num):
            data_mixlabels[rid * tau_max_per_record + i, x_dim + 1 + _l] = label_leafs[_l][rid, i]
            leaf_cum += label_leafs[_l][rid, i]
        data_mixlabels[rid * tau_max_per_record + i, -1] = _label
        if leaf_cum != _label:
            # print(leaf_cum, _label)
            delta_leaf = max(delta_leaf, abs(leaf_cum - _label))
            # print('Wrong option of Cover Tree')
            # adjust
            delta_ = _label - leaf_cum
            for _z in range(leaf_num):
                _l_z = data_mixlabels[rid * tau_max_per_record + i, x_dim + 1 + _z]
                if _l_z <= 0 or (_l_z + delta_) <= 0:
                    continue
                data_mixlabels[rid * tau_max_per_record + i, x_dim + 1 + _z] = _l_z + delta_
                break
        
        # verify
        leaf_cum_ = 0
        for _l in range(leaf_num):
            leaf_cum_ += data_mixlabels[rid * tau_max_per_record + i, x_dim + 1 + _l]
        if leaf_cum_ != _label:
            print('Wrong option of Cover Tree Again !!!')

        sc_f.append(rid * tau_max_per_record + i)    


print('Max Delta Card : ', delta_leaf)

data_mixlabels = np.array(data_mixlabels, dtype=np.float32)
data_mixlabels = data_mixlabels[sc_f]

data_mixlabels = np.unique(data_mixlabels, axis=0)

#output_file = '../data-mixlabels/face_d128_2M_trainingDataL_smallSel_CoverTree_part' + str(cid) + '-mixlabels.txt'

np.save(output_file, data_mixlabels)

