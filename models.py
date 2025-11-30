# models.py
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, hidden=256, depth=4):
        super().__init__()

        layers = []
        layers.append(nn.Linear(3, hidden))
        layers.append(nn.GELU())

        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
