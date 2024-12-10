import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, n_inputs=10):
        super(MLP, self).__init__()

        self.seq = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.BatchNorm1d(100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.Tanh(),
            nn.Linear(100, 5)
        )

    def forward(self, x):
        return self.seq(x)
