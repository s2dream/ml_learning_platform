import torch
from torch.nn.modules import Module

class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        # self.linear = torch.nn.Linear(128, 256)
        self.linear_relu_stack = torch.nn.Sequential(
            torch.nn.Linear(128, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 10),
        )
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        logit = self.softmax(output)
        return logit




