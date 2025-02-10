import torch
from torch import nn
from models.abstract import Model

class NeuralNetwork(torch.nn.Module, Model):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, num_classes, p, device):
        super().__init__()
        self.device = device
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, hidden_size1),
            nn.BatchNorm1d(hidden_size1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size1, hidden_size2),
            nn.BatchNorm1d(hidden_size2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size2, hidden_size3),
            nn.BatchNorm1d(hidden_size3),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden_size3, num_classes),
        ).to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        logits = self.layers(x)
        return logits
    
