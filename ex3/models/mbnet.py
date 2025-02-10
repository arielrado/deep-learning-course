import torch
from torch import nn
from models.abstract import Model
from torchvision.models.mobilenetv2 import MobileNetV2

class MobileNetV2Classifier(nn.Module, Model):
    def __init__(self, fixed : bool, p, device):
        super().__init__()
        self.device = device
        self.mbnet : MobileNetV2 = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True).eval()
        self.mbnet.eval()
        self.mbnet.to(self.device)

        for param in self.mbnet.parameters():
            param.requires_grad = not fixed

        self.mbnet.classifier = nn.Identity()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(1280, 4096),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024,10),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.mbnet(x)
        logits = self.layers(x)
        return logits