import torch.nn as nn
import torchvision
from torchvision import models
from torch import cat
import clip
from torchvision.models.feature_extraction import create_feature_extractor

class KGM(nn.Module):
    # Inputs an image and ouputs the prediction for the class and the projected embedding into the graph space

    def __init__(self, num_class, end_dim=128, model='resnet'):
        super(KGM, self).__init__()
        self.num_class = num_class
        self.end_dim = end_dim
        # Load pre-trained visual model
        self.deep_feature_size = None
        self.model = model
        # Load pre-trained visual model
        if model == 'resnet':
            resnet = models.resnet50(pretrained=True)
        elif model == 'vgg':
            resnet = models.vgg16(pretrained=True)
        elif model == 'clip':
            resnet, _ = clip.load("ViT-B/32")
        elif model == 'convnext':
            resnet = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
        elif model == 'vit':
            network = getattr(torchvision.models,"vit_b_16")(pretrained=True)
            self.feature_extractor = create_feature_extractor(network, return_nodes=['getitem_5'])
        
        if model != 'vit':
            self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # self.deep_feature_size = 512
        # Classifiers
        '''self.class_type = nn.Sequential(nn.Linear(2048, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(2048, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(2048, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(2048, num_class[3]))''' #TODO

        


    def forward(self, img):
        try:
            visual_emb = self.resnet(img)
        except:
            visual_emb = self.feature_extractor(img)['getitem_5']
            
        if self.deep_feature_size is None:
            # Classifier
            self.deep_feature_size = visual_emb.size(1)
            
            self.classifier1 = nn.Sequential(nn.Linear(self.deep_feature_size, self.num_class))
            # Graph space encoder
            self.nodeEmb = nn.Sequential(nn.Linear(self.deep_feature_size, self.end_dim))

            if visual_emb.is_cuda:
                self.classifier1.cuda()
                self.nodeEmb.cuda()

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        pred_class = self.classifier1(visual_emb)
        graph_proj = self.nodeEmb(visual_emb)

        return [pred_class, graph_proj]
    
    def features(self, img):
        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        pred_class = self.classifier1(visual_emb)

        return pred_class

class KGM_append(nn.Module):
    # Inputs an image and ouputs the prediction for the class and the projected embedding into the graph space

    def __init__(self, num_class, end_dim=128):
        super(KGM_append, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(2048 + end_dim, num_class))

        # Graph space encoder
        self.nodeEmb = nn.Sequential(nn.Linear(2048 + end_dim, end_dim))


    def forward(self, img):
        img, context_emb = img

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)

        emb = cat([visual_emb, context_emb.float()], 1)
        pred_class = self.classifier(emb)

        return pred_class