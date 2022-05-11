import torch.nn as nn
from torchvision import models
from torch import cat

class KGM(nn.Module):
    # Inputs an image and ouputs the prediction for the class and the projected embedding into the graph space

    def __init__(self, num_class, end_dim=128):
        super(KGM, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Classifiers
        '''self.class_type = nn.Sequential(nn.Linear(2048, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(2048, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(2048, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(2048, num_class[3]))''' #TODO

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(2048, num_class))

        # Graph space encoder
        self.nodeEmb = nn.Sequential(nn.Linear(2048, end_dim))


    def forward(self, img):

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        pred_class = self.classifier(visual_emb)
        graph_proj = self.nodeEmb(visual_emb)

        return [pred_class, graph_proj]

class KGM_append(nn.Module):
    # Inputs an image and ouputs the prediction for the class and the projected embedding into the graph space

    def __init__(self, num_class, end_dim=128):
        super(KGM_append, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Classifier
        self.classifier = nn.Sequential(nn.Linear(2048, num_class))

        # Graph space encoder
        self.nodeEmb = nn.Sequential(nn.Linear(2048, end_dim))


    def forward(self, img):
        img, context_emb = img

        visual_emb = self.resnet(img)
        visual_emb = visual_emb.view(visual_emb.size(0), -1)

        emb = cat([visual_emb, context_emb], 1)
        pred_class = self.classifier(emb)

        return pred_class