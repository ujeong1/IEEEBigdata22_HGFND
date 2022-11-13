import numpy as np
import scipy.sparse as sp
import torch
import pickle


class Hypergraph:
    def __init__(self, opt):
        self.dataset = opt.dataset


    def get_hyperedges(self):
        dirname = "data/"
        if self.dataset == "politifact":
            filename = "hypergraph_politifact.pkl"
        elif self.dataset == "gossipcop":
            filename = "hypergraph_gossipcop.pkl"

        with open(dirname + filename, 'rb') as handle:
            result = pickle.load(handle)

        return result



    def get_adj_matrix(self, hyperedges, nodes_seq):
        items, n_node, HT, alias_inputs, node_masks, node_dic = [], [], [], [], [], []

        node_list = nodes_seq
        node_set = list(set(node_list))
        node_dic = {node_set[i]: i for i in range(len(node_set))}

        rows = []
        cols = []
        vals = []
        max_n_node = len(node_set)
        max_n_edge = len(hyperedges)
        total_num_node = len(node_set)

        # num_hypergraphs can be used for batching different size of hypergraphs for training
        num_hypergraphs = 1
        for idx in range(num_hypergraphs):
            # e.g., hypergraph = [[12, 31, 111, 232],[12, 31, 111, 232],[12, 31, 111, 232] ...]
            for hyperedge_seq, hyperedge in enumerate(hyperedges):
                # e.g., hyperedge = [12, 31, 111, 232]
                for node_id in hyperedge:
                    rows.append(node_dic[node_id])
                    cols.append(hyperedge_seq)
                    vals.append(1)
            u_H = sp.coo_matrix((vals, (rows, cols)), shape=(max_n_node, max_n_edge))
            HT.append(np.asarray(u_H.T.todense()))
            alias_inputs.append([j for j in range(max_n_node)])
            node_masks.append([1 for j in range(total_num_node)] + (max_n_node - total_num_node) * [0])

        return alias_inputs, HT, node_masks
