import torch.nn as nn
from torchvision import models
import torch
import clip

class ImageSymbol(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task

    def __init__(self, num_class, model='resnet'):
        super(ImageSymbol, self).__init__()
        self.model = model
        # Load pre-trained visual model
        if model == 'resnet':
            architecture = models.resnet50(pretrained=True)
            embedding_size = 2048
        elif model == 'vgg':
            architecture = models.vgg16(pretrained=True)
            embedding_size = 25088
        elif model == 'clip':
            architecture, _ = clip.load("ViT-B/32")
            embedding_size = 512
        elif model == 'vit':
            from pytorch_pretrained_vit import ViT
            model_name = 'B_16_imagenet1k'
            architecture = ViT(model_name, pretrained=True)
            embedding_size = 768

        self.resnet = nn.Sequential(*list(architecture.children())[:-1])
            
        
        # Classifiers
        self.mlp = nn.Sequential(nn.Linear(embedding_size, num_class))

    def forward(self, img):

        if self.mode != 'clip':
            visual_emb = self.resnet(img)
        else:
            visual_emb = self.resnet.encode_image(img)

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out_type = self.mlp(visual_emb)


        return out_type