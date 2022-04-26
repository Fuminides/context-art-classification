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


class VisEncoder(nn.Module):
    '''
    Encoder that maps the visual featured from the resnet to the node2vec output size
    '''
    def __init__(self):
        super(VisEncoder, self).__init__()
        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.visual_autoencoder_l1 = nn.Sequential(nn.Linear(VISUALENCONDING_SIZE, NODE2VEC_OUTPUT), nn.ReLU())
        self.visual_autoencoder_l2 = nn.Sequential(nn.Linear(NODE2VEC_OUTPUT, VISUALENCONDING_SIZE))

    def forward(self, img):
        visual_cue = self.resnet(img).squeeze()
        l1_out = self.visual_autoencoder_l1(visual_cue)
        return self.visual_autoencoder_l2(l1_out)

    def reduce(self, img):
      visual_cue = self.resnet(img).squeeze()

      l1_out = torch.unsqueeze(self.visual_autoencoder_l1(visual_cue), 0)
      return l1_out
    
    def gen_target(self, img):
        return self.resnet(img).squeeze()
    
    def load_weights(self):
        expected_path = 'Models/Reduce/reduce_' + str(NODE2VEC_OUTPUT) + '_best_model.pth.tar'
        assert os.path.isfile(expected_path)
       
        if torch.cuda.is_available():
            checkpoint = torch.load(expected_path)
        else:
            checkpoint = torch.load(expected_path, map_location=torch.device('cpu'))
            
        if torch.cuda.is_available():
            self.load_state_dict(checkpoint['state_dict'])
        else:
            self.load_state_dict(checkpoint['state_dict']) # Does not reqquire anything else
    
    
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