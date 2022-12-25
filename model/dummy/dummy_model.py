import torch
from torch.nn.modules import Module

class DummyModel(Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.flatten = torch.nn.Flatten()
        linears = []
        linears.append(torch.nn.Linear(128, 512))
        linears.append(torch.nn.Linear(512, 1024))
        linears.append(torch.nn.Linear(1024, 512))
        linears.append(torch.nn.Linear(512, 128))
        self.linear_relu_stack = torch.nn.Sequential(
            linears[0],
            torch.nn.ReLU(),
            linears[1],
            torch.nn.ReLU(),
            linears[2],
            torch.nn.ReLU(),
            linears[3],
        )

        self.layer_norm=torch.nn.LayerNorm(128)

        linears2 = []
        linears2.append(torch.nn.Linear(128, 512))
        linears2.append(torch.nn.Linear(512, 1024))
        linears2.append(torch.nn.Linear(1024, 512))
        linears2.append(torch.nn.Linear(512, 128))
        self.linear_relu_stack_2nd = torch.nn.Sequential(
            linears2[0],
            torch.nn.ReLU(),
            linears2[1],
            torch.nn.ReLU(),
            linears2[2],
            torch.nn.ReLU(),
            linears2[3],
        )

        self.final_layer = torch.nn.Linear(128, 5)

        self.softmax = torch.nn.Softmax(dim=1)
        self.init_params(linears)
        self.init_params(linears2)

    def init_params(self, layers):
        for layer in layers:
            torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        x = output + x
        x = self.layer_norm(x)
        output = self.linear_relu_stack_2nd(x)
        output = self.final_layer(output)
        return output.to('cpu')





class DummyModel2(Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear_1st = torch.nn.Linear(128, 512)
        self.final_layer = torch.nn.Linear(512, 5)

        self.layer_norm = torch.nn.LayerNorm(512)
        self.softmax = torch.nn.Softmax(dim=1)
        self.init_params(self.linear_1st)
        self.init_params(self.final_layer)

    def init_params(self, layer):
        torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear_1st(x)
        x = self.layer_norm(x)
        output = self.final_layer(x)
        return output.to('cpu')
