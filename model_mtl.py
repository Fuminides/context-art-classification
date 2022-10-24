import torch.nn as nn
from torchvision import models
import torch
import clip

class MTL(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task

    def __init__(self, num_class, model='resnet'):
        super(MTL, self).__init__()

        
        self.model = model
        # Load pre-trained visual model
        if model == 'resnet':
            resnet = models.resnet50(pretrained=True)
            self.deep_feature_size = 2048
        elif model == 'vgg':
            resnet = models.vgg16(pretrained=True)
            self.deep_feature_size = 25088
        elif model == 'clip':
            resnet, _ = clip.load("ViT-B/32")
            self.deep_feature_size = 512


        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
            
        
        # Classifiers
        self.class_type = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[3]))

    def forward(self, img):

        if self.model != 'clip':
            visual_emb = self.resnet(img)
        else:
            visual_emb = self.resnet.encode_image(img)

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out_type = self.class_type(visual_emb)
        out_school = self.class_school(visual_emb)
        out_time = self.class_tf(visual_emb)
        out_author = self.class_author(visual_emb)

        return [out_type, out_school, out_time, out_author]