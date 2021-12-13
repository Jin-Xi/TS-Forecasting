import torch
import torch.nn as nn


class MLP(nn.Module):
    # 简单的BP网络
    def __init__(self, in_len=30, out_len=1):
        super(MLP, self).__init__()
        self.l1 = nn.Linear(in_len, int(in_len / 2))
        self.relu1 = nn.ReLU()
        self.l2 = nn.Linear(int(in_len / 2), out_len)

    def forward(self, x):
        x = self.l1(x)
        x = self.relu1(x)
        x = self.l2(x)
        return x.squeeze()
