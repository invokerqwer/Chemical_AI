#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 10:12
# @Author  : zhangbc0315@outlook.com
# @File    : handler.py
# @Software: PyCharm

import torch
import torch_scatter
from torch.nn import Module, Sequential, Linear
from torch.nn import ELU as ActFunc
import torch.nn.functional as F
import torch.nn as nn
import torch_geometric.nn as gnn

# ELU for selector


class EdgeModel(Module):
    def __init__(self, num_node_features, num_edge_features, out_features):
        super(EdgeModel, self).__init__()
        self.edge_mlp = Sequential(Linear(num_node_features + num_node_features + num_edge_features, 32),
                                   ActFunc(),
                                   # Dropout(0.5),
                                   Linear(32, out_features))

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], 1)
        return self.edge_mlp(out)


class NodeModel(Module):
    def __init__(self, num_node_features, num_edge_features_out, out_features):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Sequential(Linear(num_node_features + num_edge_features_out, 64),
                                     ActFunc(),
                                     # Dropout(0.5),
                                     Linear(64, 64))
        self.node_mlp_2 = Sequential(Linear(num_node_features + 64, 64),
                                     ActFunc(),
                                     # Dropout(0.5),
                                     Linear(64, out_features))

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        out = torch_scatter.scatter_mean(out, col, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out], dim=1)
        return self.node_mlp_2(out)


class GlobalModel(Module):
    def __init__(self, num_node_features, num_global_features, out_channels):
        super(GlobalModel, self).__init__()
        self.global_mlp = Sequential(Linear(num_global_features + num_node_features, 64),
                                     ActFunc(),
                                     # Dropout(0.3),
                                     Linear(64, out_channels))

    def forward(self, x, edge_index, edge_attr, u, batch):
        if u is None:
            out = torch_scatter.scatter_mean(x, batch, dim=0)
        else:
            out = torch.cat([u, torch_scatter.scatter_mean(x, batch, dim=0)], dim=1)
        return self.global_mlp(out)


class NPModel(nn.Module):

    def __init__(self, num_node_features, num_edge_features):
        super(NPModel, self).__init__()
        self.node_normal = gnn.BatchNorm(num_node_features)
        self.edge_normal = gnn.BatchNorm(num_edge_features)
        self.meta1 = gnn.MetaLayer(EdgeModel(num_node_features, num_edge_features, 128),
                                   NodeModel(num_node_features, 128, 32),
                                   None)
        self.meta2 = gnn.MetaLayer(EdgeModel(32, 128, 128),
                                   NodeModel(32, 128, 32),
                                   None)
        self.meta3 = gnn.MetaLayer(EdgeModel(32, 128, 128),
                                   NodeModel(32, 128, 32),
                                   None)
        self.meta8 = gnn.MetaLayer(EdgeModel(32, 128, 32),
                                   NodeModel(32, 32, 32),
                                   None)
        self.lin1 = Linear(32, 32)
        self.lin2 = Linear(32, 1)

    def forward(self, data):
        x, edge_index, e, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.node_normal(x)
        # e = self.edge_normal(e)

        x, e, _ = self.meta1(x, edge_index, e, None, batch)
        x, e, _ = self.meta2(x, edge_index, e, None, batch)
        x, e, _ = self.meta3(x, edge_index, e, None, batch)
        x, e, _ = self.meta8(x, edge_index, e, None, batch)
        y = F.elu(self.lin1(x))
        return x, self.lin2(y)


if __name__ == "__main__":
    pass
