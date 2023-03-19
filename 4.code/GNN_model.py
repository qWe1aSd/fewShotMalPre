import numpy as np
import torch
from torch import nn
from torch.nn import ReLU, Linear
from torch_geometric.nn import GATConv, GatedGraphConv
from torch_geometric.nn import global_mean_pool, global_max_pool


class GAT(nn.Module):
    def __init__(self, in_channel, out_channel, num_class):
        super(GAT, self).__init__()
        self.conv = GATConv(in_channel, out_channel)
        self.relu = ReLU()
        self.lin = Linear(out_channel * 2, num_class)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long).cuda()
        out = self.conv(x, edge_index)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        out = self.relu(out)
        out1 = global_max_pool(out, batch)
        out2 = global_mean_pool(out, batch)
        input_in = torch.cat([out1, out2], dim=-1)
        out = self.lin(input_in)
        return out


class GGNN(nn.Module):
    def __init__(self, out_channel, layers, num_class):
        super(GGNN, self).__init__()
        self.conv = GatedGraphConv(out_channel, layers)
        self.relu = ReLU()
        self.lin = Linear(out_channel * 2, num_class)

    def forward(self, x, edge_index, batch=None):
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long).cuda()
        out = self.conv(x, edge_index)
        out = torch.nn.functional.normalize(out, p=2, dim=1)
        out = self.relu(out)
        out1 = global_max_pool(out, batch)
        out2 = global_mean_pool(out, batch)
        input_in = torch.cat([out1, out2], dim=-1)
        out = self.lin(input_in)
        return out
