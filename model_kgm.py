import torch.nn as nn
from torchvision import models
from torch import cat

import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


def translate_dict(state_dict):
    for key, elem in state_dict.items():
        if 'visual_resnet' in key or 'avg_pooling_resnet' in key:
            state_dict[key.replace('visual_resnet', 'resnet')] = elem
            del state_dict[key]
    
    return state_dict

def get_gradcam(model, image, target_class_index):
        if image.shape[0] != 1:
            image = torch.unsqueeze(image, 0)

        # set the evaluation mode
        model.eval()

        # get the image from the dataloader
        img = image

        # get the most likely prediction of the model
        print('Img', img.shape)
        pred = model(img)

        # get the gradient of the output with respect to the parameters of the model
        pred[:, target_class_index].backward()

        # pull the gradients out of the model
        gradients = model.get_activations_gradient()
        # pool the gradients across the channels
        pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = model.get_activations(img).detach()

        # weight the channels by corresponding gradients
        for i in range(activations.shape[0]):
            activations[i, :, :] *= pooled_gradients[i]
            
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze().cpu()

        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)
        
        return heatmap


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

class GradCamKGM(nn.Module):
    # Inputs an image and ouputs the prediction for the class and the projected embedding into the graph space

    def __init__(self, num_class, end_dim=128):
        super(GradCamKGM, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.visual_resnet = nn.Sequential(*list(resnet.children())[0:5])
        self.avg_pooling_resnet = nn.Sequential(*list(resnet.children())[5:-1])

        self.deep_feature_size = 512
        self.classifier2 = nn.Sequential(nn.Linear(2048, self.deep_feature_size))

        self.class_type = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(self.deep_feature_size, num_class[3]))
        
        
        
        # Graph space encoder
        self.nodeEmb = nn.Sequential(nn.Linear(2048, end_dim))
      
    def forward(self, img):
        resnet_emb = self.visual_resnet(img)
        if self.training:
          h = resnet_emb.register_hook(self.activations_hook)

        resnet_emb = self.avg_pooling_resnet(resnet_emb)
        resnet_emb1 = resnet_emb.view(resnet_emb.size(0), -1)
        resnet_emb2 = self.classifier2(resnet_emb1)

        pred_type = self.class_type(resnet_emb2)
        pred_school = self.class_school(resnet_emb2)
        pred_tf = self.class_tf(resnet_emb2)
        pred_author = self.class_author(resnet_emb2)

        graph_proj = self.nodeEmb(resnet_emb1)

        return [pred_type, pred_school, pred_tf, pred_author, graph_proj]
        
      
    def get_activations_gradient(self):
        return self.gradients


    # hook for the gradients of the activations
    def activations_hook(self, grad):
        self.gradients = grad
      

    def get_activations(self, x):
      visual_emb = self.visual_resnet(x)
      
      
      return visual_emb


    def features(self, img):
        resnet_emb = self.visual_resnet(img)
        resnet_emb = self.avg_pooling_resnet(resnet_emb)
        resnet_emb1 = resnet_emb.view(resnet_emb.size(0), -1)
        resnet_emb2 = self.classifier2(resnet_emb1)

        return resnet_emb2

    
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