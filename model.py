import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from LPN_layer import LPAconv
from torch_sparse import SparseTensor, matmul
import random

class MLP(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden)
        self.fc2 = nn.Linear(hidden, out_feature)
        self.relu = nn.ReLU()
        self.dropout_rate = dropout
        self.sigmoid = torch.nn.Sigmoid()
    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.squeeze(), x.squeeze(), x.squeeze()


class GCN(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout):
        super(GCN, self).__init__()
        #self.conv1 = GCNConv(in_feature, hidden)
        #self.conv2 = GCNConv(hidden, out_feature)
        self.conv1 = GATv2Conv(in_feature, hidden)
        self.conv2 = GATv2Conv(hidden, out_feature)
        self.dropout_rate = dropout
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.sigmoid(x)
        return x.squeeze(), x.squeeze(), x.squeeze()


class FCN_LP(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout, num_edges, lpaiters, gcnnum):
        super(FCN_LP, self).__init__()
        self.edge_weight = nn.Parameter(torch.ones(num_edges))
        gc = nn.ModuleList()
        gc.append(GCNConv(in_feature, hidden))
        for i in range(gcnnum-2):
            gc.append(GCNConv(hidden, hidden))
        gc.append(GCNConv(hidden, out_feature))
        self.gc = gc
        lpn = nn.ModuleList()
        for i in range(lpaiters):
            lpn.append(LPAconv(hidden, hidden))
        self.lpn = lpn
        self.dropout_rate = dropout
        self.softmax = torch.nn.Softmax(dim=1)
        self.tanh = torch.nn.Tanh()

    def forward(self, data):
        x, edge_index, edge_attr, y = data.x, data.edge_index, data.edge_attr, data.y
        for i in range(len(self.gc)-1):            
            x = self.gc[i](x, edge_index, self.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
        out = self.gc[-1](x, edge_index, self.edge_weight)
        out = self.softmax(out)
        y_hat = out.detach()
        for i in range(len(self.lpn)):
            y_hat = self.lpn[i](x, edge_index, edge_attr, label = y_hat)
            y_hat = self.softmax(y_hat)
        return out.squeeze(), y_hat.squeeze(), x 
