import torch
from torch import nn
from models.abstract import Model

class ConvolutionalNeuralNetwork(nn.Module, Model):
    def __init__(self, p, device):
        super().__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(6272, 128),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(64, 10),
        ).to(self.device)

    def forward(self, x):
        logits = self.layers(x)
        return logits