import torch.nn as nn
from torchvision import models
import torch
import clip
import torchvision


class MTL(nn.Module):


    # Inputs an image and ouputs the predictions for each classification task

    def __init__(self, num_class: list[int], model='resnet'):
        super(MTL, self).__init__()
        self.deep_feature_size = None
        self.model = model
        # Load pre-trained visual model
        if model == 'resnet':
            resnet = models.resnet50(pretrained=True)
            self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        elif model == 'vgg':
            resnet = models.vgg16(pretrained=True)
            self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        elif model == 'clip':
            resnet, _ = clip.load("ViT-B/32")
            self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        elif model == 'convnext':
            resnet = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
            self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        elif model == 'vit':
            from torchvision.models.feature_extraction import create_feature_extractor
            network = getattr(torchvision.models,"vit_b_16")(pretrained=True)
            self.feature_extractor = create_feature_extractor(network, return_nodes=['getitem_5'])

        
        # Classifiers
        '''self.class_type = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[3]))'''

    def forward(self, img):
        if self.model == 'vit':
            visual_emb = self.feature_extractor(img)['getitem_5']
        else:
            visual_emb = self.resnet(img)
            
        if self.deep_feature_size is None:
            # Classifier
            self.deep_feature_size = visual_emb.size(1)
            
            self.classifier1 = nn.Sequential(nn.Linear(self.deep_feature_size, self.num_class[0]))
            self.classifier2 = nn.Sequential(nn.Linear(self.deep_feature_size, self.num_class[1]))
            self.classifier3 = nn.Sequential(nn.Linear(self.deep_feature_size, self.num_class[2]))
            self.classifier4 = nn.Sequential(nn.Linear(self.deep_feature_size, self.num_class[3]))
            # Graph space encoder

            if visual_emb.is_cuda:
                self.classifier1.cuda()
                self.classifier2.cuda()
                self.classifier3.cuda()
                self.classifier4.cuda()


            

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out_type = self.classifier1(visual_emb)
        out_school = self.classifier2(visual_emb)
        out_time = self.classifier3(visual_emb)
        out_author = self.classifier4(visual_emb)

        return [out_type, out_school, out_time, out_author]