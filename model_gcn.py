import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution

from torchvision import models

class GCN(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task

    def __init__(self, num_class, adj, dropout=True):
        super(GCN, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.adj = adj

        #GCN model
        self.gc1 = GraphConvolution(2048, 1028)
        self.gc2 = GraphConvolution(1028, self.final_embedding_size)
        self.dropout = dropout

        self.final_embedding_size = 128
        # Classifiers
        self.class_type = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(self.final_embedding_size, num_class[3]))

    def _GCN_forward(self, x):
        x = F.relu(self.gc1(x, self.adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, self.adj)

        return F.log_softmax(x, dim=1)

    def forward(self, img):

        visual_emb = self.resnet(img)

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out_type = self.class_type(visual_emb)
        out_school = self.class_school(visual_emb)
        out_time = self.class_tf(visual_emb)
        out_author = self.class_author(visual_emb)

        return [out_type, out_school, out_time, out_author]