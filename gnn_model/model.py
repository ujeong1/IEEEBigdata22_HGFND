import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module
from torch_geometric.nn import SAGEConv, global_max_pool
from gnn_model.layers import HyperGraphAttentionLayerSparse
import math

class PropagationEncoder(nn.Module):
    def __init__(self, args, hypergraph_model):
        super().__init__()
        self.out_channels = args.num_classes
        self.hidden_channels = args.hiddenSize
        self.in_channels = args.num_features
        self.hypergraph_model = hypergraph_model
        self.conv1 = SAGEConv(self.in_channels, self.hidden_channels)
        self.lin0 = nn.Linear(self.in_channels, self.hidden_channels)
        self.lin1 = nn.Linear(2 * self.hidden_channels, self.hidden_channels)
        self.cls = nn.Linear(self.hidden_channels, self.out_channels, bias=True)

    def forward(self, x, edge_index, HT, batch, slices):
        # Get the root node (tweet) features of each graph:
        root = (batch[1:] - batch[:-1]).nonzero(as_tuple=False).view(-1)
        root = torch.cat([root.new_zeros(1), root + 1], dim=0)
        news = x[root]
        news = self.lin0(news).relu()

        p = self.conv1(x, edge_index).relu()
        p = global_max_pool(p, batch)
        p = self.lin1(torch.cat([news, p], dim=-1)).relu()

        v = p.unsqueeze(0)
        v, e = self.hypergraph_model(v, HT)
        result = v.squeeze(0)[slices]
        return result, e

    def compute_scores(self, target):
        pred = self.cls(target)
        return F.log_softmax(pred, dim=-1)


class HGFND(nn.Module):
    def __init__(self, args):
        super(HGFND, self).__init__()
        self.hidden_size = args.hiddenSize
        self.out_channels = args.num_classes
        self.hypergraph_embedding = NewsHypergraph(args, self.hidden_size, self.out_channels)
        self.reset_parameters()

    def forward(self, nodes_embedding, HT):
        hypergraph_embedding, edge_att = self.hypergraph_embedding(nodes_embedding, HT)
        return hypergraph_embedding, edge_att

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)


class NewsHypergraph(Module):
    def __init__(self, args, initialFeatureSize, n_categories):
        super(NewsHypergraph, self).__init__()
        self.initial_feature = initialFeatureSize
        self.hidden_size = args.hiddenSize
        self.n_categories = n_categories
        self.dropout = args.dropout

        self.cls = nn.Linear(self.hidden_size, self.n_categories, bias=True)
        self.hgnn = HGNN_ATT(self.initial_feature, self.initial_feature, self.hidden_size, dropout=self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, nodes, HT):
        hypergraph, edge_att = self.hgnn(nodes, HT)  # documents are nodes and inputs
        return hypergraph, edge_att

class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.gat1 = HyperGraphAttentionLayerSparse(input_size, n_hid, dropout=self.dropout, alpha=0.2, transfer=False,
                                                   concat=True)
        self.gat2 = HyperGraphAttentionLayerSparse(n_hid, output_size, dropout=self.dropout, alpha=0.2, transfer=True,
                                                   concat=False)

    def forward(self, x, H):
        x, e = self.gat1(x, H)
        x = F.dropout(x, self.dropout, training=self.training)

        x, e = self.gat2(x, H)

        return x, e
