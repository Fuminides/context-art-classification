import torch.nn as nn
from torchvision import models
import torch
import clip
from torchvision import transforms

class SymModel(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task

    def __init__(self, num_class, model='resnet'):
        super(SymModel, self).__init__()
        self.model = model
        # Load pre-trained visual model
        if model == 'resnet':
            self.og_nmodel = models.resnet50(pretrained=True)
            embedding_size = 2048
        elif model == 'clip':
            self.og_nmodel, _ = clip.load("ViT-B/32")
            embedding_size = 512
        #self.resnet = nn.Sequential(*list(self.og_nmodel.children())[:-1])
            
        
        # Classifiers
        self.class_type = nn.Sequential(nn.Linear(embedding_size, num_class))
       

    def forward(self, img):

       
        visual_emb = self.og_nmodel(img)

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out_type = self.class_type(visual_emb)
        
        return torch.squeeze(out_type)