import math
import numpy as np

import torch.nn.functional as F
import torch.nn as nn
import torch
# torch.autograd.set_detect_anomaly(True)

# SIREN NeRF definitions


class Sine(nn.Module):
    def __init__(self, w0=30.):
        super().__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)


class SirenLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_bias=True, w0=1., is_first=False):
        super().__init__()
        self.layer = nn.Linear(input_dim, hidden_dim, bias=use_bias)
        self.activation = Sine(w0)
        self.is_first = is_first
        self.input_dim = input_dim
        self.w0 = w0
        self.c = 6
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            dim = self.input_dim
            w_std = (1 / dim) if self.is_first else (math.sqrt(self.c / dim) / self.w0)
            self.layer.weight.uniform_(-w_std, w_std)
            if self.layer.bias is not None:
                self.layer.bias.uniform_(-w_std, w_std)

    def forward(self, x):
        out = self.layer(x) 
        out = self.activation(out)
        return out

if __name__ == "__main__":
    layer = SirenLayer(128, 111)
    x = torch.randn([7, 8, 128])
    y = layer(x)
    print(x.shape)
    print(y.shape)
