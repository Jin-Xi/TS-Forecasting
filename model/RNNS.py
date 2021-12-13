import torch
import torch.nn as nn


class Net(nn.Module):
    # 简单的BP网络
    def __init__(self, in_size=30, embd_size=10, out_size=1, layer_num=3):
        super(Net, self).__init__()
        self.hidden_size = 40
        self.embd_size = embd_size
        self.in_size = in_size
        self.out_size = out_size
        self.layer_num = layer_num

        self.embd = nn.Sequential(
            nn.Linear(1, 5),
            nn.Tanh(),
            nn.Linear(5, embd_size)
        )

        self.norm = nn.LayerNorm(embd_size)

        self.gru = nn.GRU(input_size=embd_size, hidden_size=40,
                          num_layers=3, bias=True,
                          batch_first=True,
                          )
        self.l1 = nn.Linear(self.hidden_size * self.layer_num, self.out_size)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embd(x.unsqueeze(-1))
        x = self.norm(x)
        x, hidden = self.gru(x)
        hidden = hidden.permute(1, 2, 0)
        hidden = hidden.reshape(batch_size, -1)
        x = self.l1(hidden)
        return x

    def init_hidden(self, batch_size):
        return torch.zeros(self.params.gru, batch_size, self.hidden_size)
