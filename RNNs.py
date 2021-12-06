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


def plot_series(time, series, format="-", start=0, end=None):
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)


def trend(time, slope=0.):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


class time_series(Dataset):
    def __init__(self, window_size=20, pred_len=1, type='train'):
        self.window_size = window_size
        self.pred_len = pred_len
        self.type = type

        self.series, self.split_time, self.x_train, self.x_valid, self.time_train, self.time_valid = self.init_data()

        if type == 'train':
            self.data = self.x_train
            self.time = self.time_train
        if type == 'test':
            self.data = self.x_valid
            self.time = self.time_valid

    def init_data(self):
        time = np.arange(4 * 365 + 1, dtype="float32")
        baseline = 10
        amplitude = 40
        slope = 0
        noise_level = 1

        # Create the series
        series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
        # Update with noise
        series += noise(time, noise_level, seed=42)

        split_time = 3000
        time_train = time[:split_time]
        x_train = series[:split_time]
        time_valid = time[split_time:]
        x_valid = series[split_time:]
        return series, split_time, x_train, x_valid, time_train, time_valid

    def __getitem__(self, index):
        start = index
        end = index + self.window_size
        window = self.data[start:end]
        X = window[:-1, None]
        Y = X[-1]
        return X, Y

    def __len__(self):
        return len(self.data) - self.window_size


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

        self.gru = nn.GRU(input_size=10, hidden_size=40,
                          num_layers=3, bias=True,
                          batch_first=True,
                          )
        self.l1 = nn.Linear(self.hidden_size * self.layer_num, self.hidden_size)
        self.act_fn = nn.Tanh()
        self.l2 = nn.Linear(self.hidden_size, self.out_size)



    def forward(self, x):
        batch_size = x.shape[0]
        x = self.embd(x)
        x = self.norm(x)
        x, hidden = self.gru(x)
        hidden = hidden.permute(1, 2, 0)
        hidden = hidden.reshape(batch_size, -1)
        x = self.l1(hidden)
        x = self.act_fn(x)
        x = self.l2(x)
        return x

    def init_hidden(self, batch_size):
        return torch.zeros(self.params.gru, batch_size, self.hidden_size)


def train():
    net = Net(in_size=30, embd_size=10, out_size=1, layer_num=3).cuda()
    dataset = time_series(window_size=20, type='train')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    loss_fn = nn.MSELoss()


    global_step = 0
    for epoch in range(1000):
        count = 0
        total_loss = 0
        net.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            out = net(x.cuda())
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
            test(net, dataset, epoch)


def test(net, dataset, epoch):
    net.eval()
    forecast = []
    for time in range(len(dataset.series) - dataset.window_size):
        data = dataset.series[time:time + dataset.window_size - 1, None]
        data = torch.tensor(data).unsqueeze(0)
        forecast.append(net(data.cuda()))

    forecast = forecast[dataset.split_time-dataset.window_size:]
    results = torch.stack(forecast).cpu().detach().numpy().reshape(-1)

    plot_pred_and_real(results, dataset.x_valid, epoch)


def plot_pred_and_real(pred_data, real_data, epoch):
    mse = mean_squared_error(pred_data, real_data)
    # r_square = r2_score(pred_data, real_data)
    mae = mean_absolute_percentage_error(pred_data, real_data)
    print("[testing] [MSE:{}] / [MAE:{}]".format(mse, mae))
    plt.figure(figsize=(10, 6))
    plt.grid(True)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(real_data, "blue", label="real data")
    plt.plot(pred_data, "green", label='pred data')
    plt.title("[MSE:{}] / [MAE:{}]".format(mse, mae))
    plt.legend()
    # plt.savefig('./animation/LSTM/'+str(epoch)+'.jpg')
    plt.show()


if __name__ == "__main__":
    train()
    # net = Net()
    # input = torch.randn(32, 20, 1)
    # output = net(input)
    # pass
