import os

import numpy as np

#from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
if torch.cuda.is_available():
    try:
        from torch_geometric.nn import GCNConv
    except ModuleNotFoundError:
        print('Pytorch geometric not found, proceeding...')
else:
    print('Unable to import Pytorch geometric (CPU only not supported)')


from torchvision import models

NODE2VEC_OUTPUT = 128
VISUALENCONDING_SIZE = 2048


  
    
class GCN(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task
    
    def __init__(self, in_channels, hidden_channels, num_class, target_class='all'):
        super(GCN, self).__init__()
        
        self.hidden_size = hidden_channels
        ntype, nschool, ntime, nauthor = num_class
         
        #GCN model
        self.gc1 = GCNConv(in_channels, self.hidden_size)

        # GCN convs
        self.gc_type = GCNConv(self.hidden_size, ntype)
        self.gc_nschool = GCNConv(self.hidden_size, nschool)
        self.gc_ntime = GCNConv(self.hidden_size, ntime)
        self.gc_nauthor = GCNConv(self.hidden_size, nauthor)

        # Classifiers
        self.class_type = nn.Sequential(nn.Linear(ntype, num_class[0]), nn.Softmax())
        self.class_school = nn.Sequential(nn.Linear(nschool, num_class[1]), nn.Softmax())
        self.class_tf = nn.Sequential(nn.Linear(ntime, num_class[2]), nn.Softmax())
        self.class_author = nn.Sequential(nn.Linear(nauthor, num_class[3]), nn.Softmax())

        self.target_class = target_class

    def forward(self, x, edge_index):
        if isinstance(edge_index, list):
            adjs = edge_index 
            edge_index, e_id, size = adjs[0]
            x_target = x[:size[1]]
            x = F.relu(self.gc1((x, x_target), edge_index))
            edge_index, e_id, size = adjs[1]
            
        else:
        
            x = F.relu(self.gc1(x, edge_index))
            x = F.dropout(x, training=self.training)

        if self.target_class == 'type':
            graph_emb = self.gc_type(x, edge_index)
            out_type = self.class_type(graph_emb)

            return out_type
        elif self.target_class == 'school':
            graph_emb = self.gc_nschool(x, edge_index)
            out_school = self.class_school(graph_emb)

            return out_school

        elif self.target_class == 'time':
            graph_emb = self.gc_ntime(x, edge_index)
            out_time = self.class_tf(graph_emb)

            return out_time
        elif self.target_class == 'author':
            graph_emb = self.gc_nauthor(x, edge_index)
            out_author = self.class_author(graph_emb)

            return out_author

        elif self.target_class == 'all':
            graph_emb = self.gc_type(x, edge_index)
            out_type = self.class_type(graph_emb)
            graph_emb = self.gc_nschool(x, edge_index)
            out_school = self.class_school(graph_emb)
            graph_emb = self.gc_ntime(x, edge_index)
            out_time = self.class_tf(graph_emb)
            graph_emb = self.gc_nauthor(x, edge_index)
            out_author = self.class_author(graph_emb)

            return [out_type, out_school, out_time, out_author]