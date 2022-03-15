import os

import numpy as np

#from torch_geometric.data import Data

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GCNConv

from scipy.sparse import coo_matrix

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
        return self.visual_autoencoder_l1(self.visual_autoencoder_l1(self.resnet(img).squeeze()))
    
    def gen_target(self, img):
        return self.resnet(img).squeeze()
    
    def load_weights(self):
        expected_path = 'Models/Reduce/reduce_' + str(NODE2VEC_OUTPUT) + '_best_model.pth.tar'
        assert os.path.isfile(expected_path)
       
        checkpoint = torch.load(expected_path)
        self.load_state_dict(checkpoint['state_dict'])
    
class GCN(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task
    
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        self.final_embedding_size = out_channels
        self.hidden_size = hidden_channels
        
        # Autoencoders for visual features
        if NODE2VEC_OUTPUT != VISUALENCONDING_SIZE:
            self.vis_reducer = VisEncoder()
            # Load the model
            if torch.cuda.is_available():
                self.vis_reducer = self.vis_reducer.cuda()
            self.vis_reducer.load_weights()

         
        #GCN model
        self.gc1 = GCNConv(NODE2VEC_OUTPUT, self.hidden_channels)
        self.gc2 = GCNConv(self.hidden_size, self.final_embedding_size)
        self.dropout = dropout

        # Classifiers
        self.class_type = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[3]))

    def _GCN_forward(self, x):
        x = F.relu(self.gc1(x, self.adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj)

        return x # F.log_softmax(x, dim=1) # Softmax needed?

    def forward(self, img):

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        
        if NODE2VEC_OUTPUT != VISUALENCONDING_SIZE:
            visual_emb = self.visual_autoencoder_l1(visual_emb)
            
        graph_emb = self._GCN_forward(visual_emb)
        
        out_type = self.class_type(graph_emb)
        out_school = self.class_school(graph_emb)
        out_time = self.class_tf(graph_emb)
        out_author = self.class_author(graph_emb)

        return [out_type, out_school, out_time, out_author]