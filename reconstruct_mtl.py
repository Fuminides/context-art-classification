import torch.nn as nn
from torchvision import models

class RMTL(nn.Module):
    # Inputs an image and ouputs the predictions for each classification task

    def __init__(self, num_class):
        super(RMTL, self).__init__()

        # Load pre-trained visual model
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])

        # Classifiers
        self.class_type = nn.Sequential(nn.Linear(128, num_class[0]))
        self.class_school = nn.Sequential(nn.Linear(128, num_class[1]))
        self.class_tf = nn.Sequential(nn.Linear(128, num_class[2]))
        self.class_author = nn.Sequential(nn.Linear(128, num_class[3]))

        # Encoder and decoder
        self.encoder = nn.Linear(2048, 128)
        self.decoder = nn.Linear(128, 2048)

    def forward(self, img):

        visual_emb0 = self.resnet(img)
        visual_emb0 = visual_emb0.view(visual_emb0.size(0), -1)
        visual_emb = self.encoder(visual_emb0)

        out_type = self.class_type(visual_emb)
        out_school = self.class_school(visual_emb)
        out_time = self.class_tf(visual_emb)
        out_author = self.class_author(visual_emb)

        reconstructed_visual = self.decoder(visual_emb)


        return [out_type, out_school, out_time, out_author, reconstructed_visual, visual_emb]