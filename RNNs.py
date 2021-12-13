from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
import pandas as pd

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from tqdm import tqdm

from utils.plot_pred_real import plot_pred_and_real
from utils.test_model import test_RNNS as test
from data.datasets.time_series import time_series


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


def train(model, input_len, output_len):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_dataset = time_series(window_size=input_len+output_len, input_len=input_len, pred_len=output_len, type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    vali_dataset = time_series(window_size=input_len+output_len, input_len=input_len, pred_len=output_len, type='test')
    vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)

    global_step = 0
    for epoch in range(1000):
        count = 0
        total_loss = 0
        net.train()
        for x, y in train_dataloader:
            optimizer.zero_grad()
            out = model(x.cuda())
            loss = loss_fn(out, y.cuda())
            loss.backward()

            count += 1
            global_step += 1
            total_loss += loss.item()

            if count % 10 == 0:
                total_loss /= count
                count = 0
                # scheduler.step(total_loss)
                print('[training] [epoch:{}/{} , iter:{}, total_loss:{}]'.format(epoch, 1000, global_step, total_loss))
                total_loss = 0

            optimizer.step()

        if epoch % 5 == 0:
            test(net, epoch, vali_dataloader)


if __name__ == "__main__":
    input_len, output_len = 5, 1
    net = Net(in_size=input_len, embd_size=40, out_size=output_len, layer_num=3).cuda()
    train(net, input_len, output_len)