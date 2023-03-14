import torch.nn as nn
from torchvision import models
from torch import cat

class KGM(nn.Module):
    # Inputs an image and ouputs the prediction for the class and the projected embedding into the graph space

    def __init__(self, num_class, end_dim=128, multi_task=False):
        super(KGM, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.deep_feature_size = 20
        self.classifier1 = nn.Sequential(nn.Linear(2048, self.deep_feature_size))
        self.multi_task = multi_task
        
        if self.multi_task:            
            # Classifiers
            self.class_type = nn.Sequential(nn.Linear(2048, num_class[0]))
            self.class_school = nn.Sequential(nn.Linear(2048, num_class[1]))
            self.class_tf = nn.Sequential(nn.Linear(2048, num_class[2]))
            self.class_author = nn.Sequential(nn.Linear(2048, num_class[3]))

        # Classifier
        else:
            self.classifier2 = nn.Sequential(nn.Linear(self.deep_feature_size, num_class))

        # Graph space encoder
        self.nodeEmb = nn.Sequential(nn.Linear(2048, end_dim))


    def forward(self, img):

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)

        if self.multi_task:
            pred_type = self.class_type(visual_emb)
            pred_school = self.class_school(visual_emb)
            pred_tf = self.class_tf(visual_emb)
            pred_author = self.class_author(visual_emb)
            graph_proj = self.nodeEmb(visual_emb)

            return [pred_type, pred_school, pred_tf, pred_author, graph_proj]

        else:
            pred_class = self.classifier1(visual_emb)
            pred_class = self.classifier2(pred_class)
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