import torch
from torch.nn.modules import Module

class DummyModel(Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        linears = []
        linears.append(torch.nn.Linear(128, 512))
        linears.append(torch.nn.Linear(512, 1024))
        linears.append(torch.nn.Linear(1024, 512))
        linears.append(torch.nn.Linear(512, 5))
        self.linear_relu_stack = torch.nn.Sequential(
            linears[0],
            torch.nn.ReLU(),
            linears[1],
            torch.nn.ReLU(),
            linears[2],
            torch.nn.ReLU(),
            linears[3],
        )

        self.softmax = torch.nn.Softmax(dim=1)
        self.init_params(linears)

    def init_params(self, layers):
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        logit = self.softmax(output)
        return logit




