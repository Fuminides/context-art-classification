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
        elif model == 'vgg':
            self.og_nmodel = models.vgg16(pretrained=True)
            embedding_size = 25088
        elif model == 'clip':
            self.og_nmodel, _ = clip.load("ViT-B/32")
            embedding_size = 512
        elif model == 'vit':
            from pytorch_pretrained_vit import ViT
            model_name = 'B_16_imagenet1k'
            self.og_nmodel = ViT(model_name, pretrained=True)
            self.tfms = transforms.Compose([transforms.Resize(self.og_nmodel.image_size)])

            embedding_size = 768

        self.resnet = nn.Sequential(*list(self.og_nmodel.children())[:-1])
            
        
        # Classifiers
        self.class_type = nn.Sequential(nn.Linear(embedding_size, num_class))
       

    def forward(self, img):

        if self.model == 'vit':
            img = self.tfms(img)

        if self.model != 'clip':
            visual_emb = self.resnet(img)
        else:
            visual_emb = self.resnet.encode_image(img)

        visual_emb = visual_emb.view(visual_emb.size(0), -1)
        out_type = self.class_type(visual_emb)
        
        return torch.squeeze(out_type)