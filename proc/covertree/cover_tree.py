# Spherical Range Query Cardinality Estimation
# Cover Tree based Model
# Inner node:   representations
# leaf node:    learn good threshold partition and prediction


from __future__ import division

import numpy as np
from collections import defaultdict
import operator
import itertools
import random
from heapq import heappush, heappop
import pickle

from timeit import default_timer as timer

import tensorflow as tf

class CoverTree(object):
    # define children distances
    class _lazy_child_dist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base

        def __missing__(self, i):
            self[i] = value = self.b ** i
            return value

    class _lazy_hier_dist(dict):
        def __init__(self, base, *a, **kw):
            dict.__init__(self, *a, **kw)
            self.b = base
        def __missing__(self, i):
            self[i] = value = self.b ** (i + 1) / (self.b - 1)
            return value

    def __init__(self, originalData, distance,
                 leaf_data_file_prefix, cover_tree_dir,
                 leafsize=100000, base=2):
        # construct a cover tree
        self.originalData = originalData
        self.originalDataNum = self.originalData.shape[0]
        # assume Xs are vectors
        self.originalDataDim = self.originalData.shape[1]
        self.distance = distance
        self.leafsize = leafsize
        if self.leafsize < 1:
            raise ValueError("Leaf node must contain at least 1 data")

        self._child_d = CoverTree._lazy_child_dist(base)
        self._heir_d = CoverTree._lazy_hier_dist(base)

        # save original data of all leaf nodes into file
        self.leaf_data_file_prefix = leaf_data_file_prefix

        # save cover tree structure into a dir
        self.cover_tree_dir = cover_tree_dir

    class _Node(object):
        # define a node of CoverTree
        pass

    class _InnerNode(_Node):
        def __init__(self, ctr_idx, level, radius, children, inner_node_id):
            # originalData[ctr_idx] belong to this inner node
            # children are within _dist[level] of originalData[ctr_idx]
            # level is the current level of this inner node in CT
            # in this inner node, a representation is generated
            self.ctr_idx = ctr_idx
            self.level = level
            self.radius = radius
            # self.children = children
            # self.num_children = sum([c.num_children for c in children])

            self._set_children(children)

            self.inner_node_id = inner_node_id

        def _set_children(self, children):
            self.children = children
            self.num_children = sum([c.num_children for c in children])

        def __repr__(self):
            return ("<InnerNode: ctr_idx=%d, level=%d (radius=%f), "
                    "len(children)=%d, num_children=%d>" %
                    (self.ctr_idx, self.level,
                     self.radius, len(self.children), self.num_children))


    class _LeafNode(_Node):
        def __init__(self, idx, ctr_idx, radius, leaf_node_id, data_file_prefix):
            self.idx = idx
            self.ctr_idx = ctr_idx
            self.radius = radius
            self.num_children = len(idx)

            self.data_file_prefix = data_file_prefix

            # leaf node ID
            self.leaf_node_id = leaf_node_id

        def __repr__(self):
            return ('<LeafNode: idx=%s, ctr_idx=%d, radius=%f>' %
                    (len(self.idx), self.ctr_idx, self.radius))

        def _set_leafID(self, leaf_id):
            self.leaf_id = leaf_id

        # save original data **IDs** of leaf nodes
        def _save_original_data_IDS(self):
            data_file = self.data_file_prefix + '_LEAF_ID_' + str(self.leaf_node_id) + '.npy'
            np.save(data_file, np.array(self.idx, dtype=np.uint32))

        def _get_leaf_center_info(self):
            return [self.leaf_node_id, self.ctr_idx, self.radius]

    def _build(self):
        # construct the Cover Tree
        child_d = self._child_d
        heir_d = self._heir_d

        leaf_centers = []
        def split_with_dist(dmax, Dmax, pts_p_ds):
            # split the points in two lists
            # one closer than dmax to p
            # the other one up tp DMax away
            near_p_ds, far_p_ds = [], []
            new_pts_len = 0
            for i in range(len(pts_p_ds)):
                idx, dist_p = pts_p_ds[i]
                if dist_p <= dmax:
                    near_p_ds.append((idx, dist_p))
                elif dist_p <= Dmax:
                    far_p_ds.append((idx, dist_p))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_p_ds, far_p_ds

        def split_without_dist(q_idx, dmax, Dmax, pts_p_ds):
            near_q_ds, far_q_ds = [], []
            new_pts_len = 0
            for i in range(len(pts_p_ds)):
                idx, _ = pts_p_ds[i]
                dist_p = self.distance(self.originalData[q_idx], self.originalData[idx])
                if dist_p <= dmax:
                    near_q_ds.append((idx, dist_p))
                elif dist_p <= Dmax:
                    far_q_ds.append((idx, dist_p))
                else:
                    pts_p_ds[new_pts_len] = pts_p_ds[i]
                    new_pts_len += 1
            pts_p_ds[:] = pts_p_ds[:new_pts_len]

            return near_q_ds, far_q_ds

        def _construct(p_idx, near_p_ds, far_p_ds, i):
            # main construct CT
            # if reach the leaf node
            if len(near_p_ds) + len(far_p_ds) <= self.leafsize:
                idx = [ii for (ii, d) in itertools.chain(near_p_ds,
                                                         far_p_ds)]
                radius = max(d for (ii, d) in itertools.chain(near_p_ds,
                                                               far_p_ds,
                                                               [(0.0, -1)]))
                node = CoverTree._LeafNode(idx, p_idx, radius, self.leaf_node_IDS, self.leaf_data_file_prefix)
                # save original data of leaf node
                print('Node L : ', self.leaf_node_IDS)
                node._save_original_data_IDS()
                leaf_centers.append(node._get_leaf_center_info())
                self.leaf_node_IDS += 1

                # node._set_leafID(len(self.leaf_vec))    # this may involve some problems !!!
                # self.leaf_vec.append(node)

                return node, []
            else:
                nearer_p_ds, so_so_near_p_ds = split_with_dist(child_d[i - 1],
                                                               child_d[i], near_p_ds)

                p_im1, near_p_ds = _construct(p_idx, nearer_p_ds, so_so_near_p_ds, i-1)

                if not near_p_ds:
                    return p_im1, far_p_ds
                else:
                    children = [p_im1]
                    while near_p_ds:
                        # random select one center
                        random.seed(20)
                        q_idx, _ = random.choice(near_p_ds)
                        near_q_ds, far_q_ds = split_without_dist(q_idx,
                                                child_d[i-1], child_d[i], near_p_ds)
                        near_q_ds2, far_q_ds2 = split_without_dist(q_idx,
                                                child_d[i-1], child_d[i], far_p_ds)
                        near_q_ds += near_q_ds2
                        far_q_ds += far_q_ds2

                        q_im1, unused_q_ds = _construct(q_idx,
                                            near_q_ds, far_q_ds, i - 1)
                        children.append(q_im1)
                        # deal with unused points
                        new_near_p_ds, new_far_p_ds = split_without_dist(p_idx,
                                            child_d[i], child_d[i+1], unused_q_ds)
                        near_p_ds += new_near_p_ds
                        far_p_ds += new_far_p_ds

                    # Construct an inner node
                    p_i = CoverTree._InnerNode(p_idx, i, heir_d[i], children,
                                           self.inner_node_IDs)
                    print('Node I : ', self.inner_node_IDs)
                    self.inner_node_IDs += 1
                    self.inner_node_mapping_IDS += len(children)

                    return p_i, far_p_ds


        # count the IDS of inner and leaf nodes
        self.inner_node_IDs, self.leaf_node_IDS = 0, 0
        # record the position of the inner node in the mapping matrix
        self.inner_node_mapping_IDS = 0

        if self.originalDataNum == 0:
            self.root = CoverTree._LeafNode(idx=[], ctr_idx=-1, radius=0,
                                        leaf_node_id=self.leaf_node_IDS,
                                        data_file_prefix=self.leaf_data_file_prefix)
            self.root._save_original_data_IDS()
            self.leaf_node_IDS += 1

        else:
            random.seed(20)
            p_idx = random.randrange(self.originalDataNum)
            #print('random array', len(p_idx))
            near_p_ds = [(j, self.distance(self.originalData[p_idx], self.originalData[j]))
                                    for j in np.arange(self.originalDataNum)]
            far_p_ds = []
            try:
                maxdist = 2 * max(near_p_ds, key=operator.itemgetter(1))[1]
            except ValueError:
                maxdist = 1
            maxlevel = 0
            while maxdist > child_d[maxlevel]:
                maxlevel += 1

            self.root, unused_p_ds = _construct(p_idx, near_p_ds,
                                                far_p_ds, maxlevel)

            leaf_centers = np.array(leaf_centers)
            np.save(self.leaf_data_file_prefix + '_CENTERS.npy', leaf_centers)
            print('Number of unused points is ', len(unused_p_ds))

    def _print(self):
        # print the built Cover Tree
        def print_node(node, indent):
            if isinstance(node, CoverTree._LeafNode):
                print('-' * indent, node)
            else:
                print('-' * indent, node)
                for child in node.children:
                    print_node(child, indent + 1)
        print_node(self.root, 0)

    def _serialize_CT(self):
        '''
        Include a configuration file (each row is a structure ID)
        :return: None
        '''
        self.structure_arr = ""
        def _serialize(node):
            if isinstance(node, CoverTree._LeafNode):
                self.structure_arr += 'LEAF_' + str(node.leaf_node_id) + " "
                node_file = os.path.join(self.cover_tree_dir, 'LEAF_' + str(node.leaf_node_id))
                pickle.dump(node, open(node_file, 'wb'))
            else:
                node_file = os.path.join(self.cover_tree_dir, 'INNER_' + str(node.inner_node_id))
                pickle.dump(node, open(node_file, 'wb'))
                children_num = len(node.children)
                self.structure_arr += str(children_num) + " "
                for child_id in range(children_num):
                    _serialize(node.children[child_id])

        _serialize(self.root)

        # save structure configuration file
        config_file = os.path.join(self.cover_tree_dir, 'CT_structure')
        f = open(config_file, 'w')
        f.write(self.structure_arr)

    def _deserialize_CT(self):
        config_file = os.path.join(self.cover_tree_dir, 'CT_structure')
        f = open(config_file, 'r')
        self.structure_arr = f.read()

        def _deserialize(structure_arrs, structure_id):
            node_id = structure_arrs[structure_id].split('_')[0]
            if node_id == 'LEAF':
                node_file = os.path.join(self.cover_tree_dir, structure_arrs[structure_id])
                node = pickle.load(open(node_file, 'rb'))
                return node
            else:
                node_file = os.path.join(self.cover_tree_dir, structure_arrs[structure_id])
                node = pickle.load(open(node_file, 'rb'))
                structure_id += 1
                children_num_ = int(structure_arrs[structure_id])
                children_num = len(node.children)
                if children_num != children_num_:
                    raise ValueError('Wrong Load Cover Tree Structure')
                for child_id in range(children_num_):
                    pass
                return node

        self.root = _deserialize(self.structure_arr.split(), 0)


    def _query_ball_point(self, x, r, eps=0):
        def traverse_checking(node):
            d_x_node = self.distance(x, self.originalData[node.ctr_idx])
            min_distance = max(0.0, d_x_node - node.radius)
            max_distance = d_x_node + node.radius
            if min_distance > r * (1 + eps):
                return []
            elif max_distance < r * (1 + eps):
                return traverse_no_checking(node)
            elif isinstance(node, CoverTree._LeafNode):
                # insert x into Leaf node
                _l = list(i for i in node.idx
                          if self.distance(x, self.originalData[i]) <= r)
                return _l
            else:
                return list(itertools.chain.from_iterable(
                            traverse_checking(child)
                            for child in node.children))

        def traverse_no_checking(node):
            if isinstance(node, CoverTree._LeafNode):
                return node.idx
            else:
                return list(itertools.chain.from_iterable(
                    traverse_no_checking(child) for child in node.children))

        return traverse_checking(self.root)

    def query_ball_point(self, x, r, eps=0):
        # query spherical range
        x = np.asarray(x)
        if self.pt_shape and x.shape[-len(self.pt_shape):] != self.pt_shape:
            raise ValueError("Search error ...")
        if len(x.shape) == 1:
            # one query
            return self._query_ball_point(x, r, eps)
        else:
            if self.pt_shape:
                retshape = x.shape[:-len(self.pt_shape)]
            else:
                retshape = x.shape
            result = np.empty(retshape, dtype=np.object)
            for c in np.ndindex(retshape):
                result[c] = self._query_ball_point(x[c], r, eps=eps)
            return result

    ''' The following is to insert training / testing data in order to 
        fill a mask matrix
        OR train or infer data with CoverTree according to the mask matrix
    '''
    #def _infer_ball_point_fill_map(self, x, r, data_idx, batch_size, batch_num, mapping, eps=0):
    def _infer_ball_point_fill_map(self, x, r, data_idx, mapping, eps=0):
        '''
        :param x:
        :param r:
        :param data_idx:
        :param mapping: a matrix of type np.uint8
        :param eps:
        :return:
        '''
        def traverse_checking(node):
            d_x_node = self.distance(x, self.originalData[node.ctr_idx])
            min_distance = max(0.0, d_x_node - node.radius)
            max_distance = d_x_node + node.radius
            if min_distance > r / (1. + eps):
                return
            elif max_distance < r * (1. + eps):
                return traverse_no_checking(node)
            elif isinstance(node, CoverTree._LeafNode):
                # insert the id of x (i.e., training or testing data)
                node._insert_trainID(data_idx)
                # prediction_leaf = node._construct_predictions()
                return
            else:
                # check valid children
                valid_children = []
                for cid, child in enumerate(node.children):
                    dist_ = self.distance(x, self.originalData[child.ctr_idx])
                    min_dist_ = max(0.0, dist_ - child.radius)
                    if min_dist_ <= r / (1. + eps):
                        valid_children.append(cid)
                # insert mapping information of assigning children in current node
                #node._insert_data(data_idx, batch_size, batch_num, valid_children)
                for v_cid in valid_children:
                    mapping[data_idx, node.inner_mapping_id + v_cid] = 1
                # go to children
                for child in node.children:
                    traverse_checking(child)
                return

        # def _construct_predictions(self, out, Xs, taus, hidden_layers_tau, hidden_layers_p, part_num, tau_max):
        def traverse_no_checking(node):
            if isinstance(node, CoverTree._LeafNode):
                node._insert_trainID(data_idx)
                return
            else:
                # all children are valid
                valid_children = [cid for cid, child in enumerate(node.children)]
                #node._insert_data(data_idx, batch_size, batch_num, valid_children)
                for v_cid in valid_children:
                    mapping[data_idx, node.inner_mapping_id + v_cid] = 1
                # go to children
                for child in node.children:
                    traverse_no_checking(child)
                return

        return traverse_checking(self.root)

    def infer_ball_point_fill_map(self, X, Taus, mapping, eps=0):
        '''
        param X: a list of vector, i.e., matrix
        :param Taus: a list of thresholds, i.e., vectors=
        :param mapping: a dispatcher matrix
        :param eps:
        :return:
        '''
        data_num = X.shape[0]
        for rid in range(data_num):
            self._infer_ball_point_fill_map(X[rid], Taus[rid], rid, mapping, eps)
        return

