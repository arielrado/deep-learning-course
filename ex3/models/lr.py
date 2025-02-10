import torch
from models.abstract import Model

class LogisticRegression(torch.nn.Module, Model):
    def __init__(self, input_size, num_classes, device):
        super().__init__()
        self.device = device
        self.linear = torch.nn.Linear(input_size, num_classes, bias = True, device=self.device)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1).requires_grad_().to(self.device)
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        logits = self.linear(x)
        return logits
    
    def predict(self, x):
        x.to(self.device)
        outputs = self(x)
        _, predicted = torch.max(outputs, 1)
        return predicted

    def evaluate(self, x, y):
        x.to(self.device)
        y = y.to(self.device)
        predicted = self.predict(x)
        correct = (predicted == y).sum().item()
        return correct / y.size(0)
