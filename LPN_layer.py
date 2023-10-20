from typing import Callable, Optional
from torch_geometric.typing import Adj, OptTensor

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
import math

class LPAconv(MessagePassing):
    def __init__(self, in_channels, out_channels, heads = 2, negative_slope = 0.2, dropout = 0.):
        super(LPAconv, self).__init__(aggr='sum')#
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_l = torch.nn.Linear(self.in_channels, self.out_channels)
        self.lin_r = self.lin_l
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.heads = heads
        self.linear = nn.Linear(2*out_channels, 2)
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)


    def forward(
            self, x: Tensor, edge_index: Adj, edge_attr : Tensor, label : Tensor,
            post_step: Callable = lambda y: y.clamp_(0., 1.)
    ) -> Tensor:
        
        edge_index = SparseTensor.from_edge_index(edge_index)
        x_l = self.lin_l(x)  #W_l*h_i [N,H*C]   
        x_r = self.lin_r(x)  #W_r*h_j
        y_out  = self.propagate(edge_index, x=(x_l, x_r), label = label, edge_attr = edge_attr)  #[N,H,C]
        return y_out 

    def message(self, x_i: Tensor, x_j: Tensor, label_i, label_j, index, ptr, size_i, edge_attr) -> Tensor:

        alpha = self.linear(torch.cat((x_i, x_j), dim=-1))
        alpha = torch.tanh(alpha)
        y_out = label_j * alpha  #[E,H,C]  
        return y_out
   