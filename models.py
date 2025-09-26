import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet=models.resnet50(pretrained=True)
        modules=list(resnet.children())[:-1]
        self.resnet=nn.Sequential(*modules)
        self.linear=nn.Linear(resnet.fc.in_feature,embed_size)
        self.bn=nn.BatchNorm1d(embed_size)
    def forward(self,image):
        with torch.no_grad():
            features=self.resnet(image).squeeze()
            return self.bn(self.linear(features))
        