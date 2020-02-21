import numpy as np
import tensorflow as tf

''' The file is to dispatch data to the following children in CTNet
'''

class Dispatcher(object):
    '''
        The class is for assigning train/testing data to the following children nodes
    '''
    def __init__(self, X_map, num_children_nodes):
        '''

        :param X_map: a mask to assign, and it is **fixed** !!!
        :param num_children_nodes: how many children nodes to be assigned to
        '''
        self.gate = X_map
        assert self.gate.shape[1] == num_children_nodes
        self.num_children_nodes = num_children_nodes

        where = tf.to_int32(tf.where(tf.transpose(self.gate) > 0))
        self._expert_index, self._batch_index = tf.unstack(where, num=2, axis=1)
        self._part_sizes_tensor = tf.reduce_sum(tf.to_int32(self.gate > 0), [0])
        self._nonzeros_gates = tf.gather(
            tf.reshape(self.gate, [-1]), self._batch_index * self.num_children_nodes + self._expert_index
        )

    def dispatch(self, X, Tau, Y):
        X = tf.gather(X, self._batch_index)
        _X_s = tf.split(X, self._part_sizes_tensor, 0, num=self.num_children_nodes)

        Tau = tf.gather(Tau, self._batch_index)
        _Tau_s = tf.split(Tau, self._part_sizes_tensor, 0, num=self.num_children_nodes)

        Y = tf.gather(Y, self._batch_index)
        _Y_s = tf.split(Y, self._part_sizes_tensor, 0, num=self.num_children_nodes)

        return _X_s, _Tau_s, _Y_s

    def dispatch_(self, Repr, X, Tau, Y):
        Repr = tf.gather(Repr, self._batch_index)
        _Repr_s = tf.split(Repr, self._part_sizes_tensor, 0, num=self.num_children_nodes)

        X = tf.gather(X, self._batch_index)
        _X_s = tf.split(X, self._part_sizes_tensor, 0, num=self.num_children_nodes)

        Tau = tf.gather(Tau, self._batch_index)
        _Tau_s = tf.split(Tau, self._part_sizes_tensor, 0, num=self.num_children_nodes)

        Y = tf.gather(Y, self._batch_index)
        _Y_s = tf.split(Y, self._part_sizes_tensor, 0, num=self.num_children_nodes)

        return _Repr_s, _X_s, _Tau_s, _Y_s


    def combines(self, children_labels_vec):
        stitched = tf.concat(children_labels_vec, 0)
        stitched = tf.expand_dims(self._nonzeros_gates, 1)
        combined = tf.unsorted_segment_sum(stitched, self._batch_index,
                                                tf.shape(self.gate)[0])
        return combined
