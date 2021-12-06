from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

import logging
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.plot_pred_real import plot_pred_and_real
from utils.build_fake_data import *


torch.manual_seed(888)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('DeepAR.Net')


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


class time_series(torch.utils.data.Dataset):
    """Some Information about Time_Series"""
    def __init__(self, data: np.array, input_window, output_window, type='train'):
        super(time_series, self).__init__()
        self.raw_data = data
        self.input_window = input_window
        self.output_window = output_window

        if type == 'train':
            self.data = torch.tensor(data[:int(len(data) * 0.6)])
            self.moving_stride = 1
        elif type == 'test':
            self.data = torch.tensor(data[int(len(data) * 0.6):])
            self.moving_stride = output_window

    def __getitem__(self, index):
        index = index * self.moving_stride
        input_window = self.data[index:index+self.input_window]
        output_window = self.data[index+self.input_window:index+self.input_window+self.output_window]
        return input_window.unsqueeze(-1), output_window.unsqueeze(-1)

    def __len__(self):
        return (len(self.data) - self.input_window) // self.moving_stride

#####################################################
#
#   定义网络：
#           应该使用encoder 和 decoder 结构
#
#####################################################


class Encoder(nn.Module):
    def __init__(self, input_size=1, embd_size=10, hidden_size=40, num_layers=3):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # self.casual_conv1d = nn.Conv1d(in_channels=params.num_class, out_channels=params.num_class, )
        self.embedding = nn.Linear(input_size, embd_size)
        self.rnn = nn.GRU(input_size=embd_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bias=True,
                           batch_first=False,
                           dropout=0.1)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def forward(self, x):
        # x [seq_len, batch_size, class_num]
        output = self.embedding(x)
        output, hidden = self.rnn(output)
        return output, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


# noinspection PyStatementEffect
class Decoder(nn.Module):
    def __init__(self, input_size=1, embd_size=10, hidden_size=40, num_layers=3):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.embd_size = embd_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Linear(input_size, embd_size)
        self.rnn = nn.GRU(input_size=embd_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bias=True,
                           batch_first=False,
                           dropout=0.1)

        # initialize LSTM forget gate bias to be 1 as recommanded by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for names in self.rnn._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.rnn, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

        self.distribution_mu = nn.Sequential(
            nn.Linear(hidden_size*num_layers, hidden_size*num_layers),
            nn.ReLU(),
            nn.Linear(hidden_size*num_layers, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.distribution_presigma = nn.Sequential(
            nn.Linear(hidden_size*num_layers, hidden_size*num_layers),
            nn.ReLU(),
            nn.Linear(hidden_size*num_layers, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.distribution_sigma = nn.Softplus()


    def forward(self, x, hidden):
        lstm_input = self.embedding(x)
        output, hidden = self.rnn(lstm_input, hidden)
        hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        pre_sigma = self.distribution_presigma(hidden_permute)
        sigma = self.distribution_sigma(pre_sigma)
        mu = self.distribution_mu(hidden_permute)
        return torch.squeeze(mu), torch.squeeze(sigma), hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
    distribution = torch.distributions.normal.Normal(torch.squeeze(mu), torch.squeeze(sigma))
    likelihood = distribution.log_prob(torch.squeeze(labels))
    return -torch.mean(likelihood)


def train():

    encoder = Encoder(input_size=1, embd_size=10, hidden_size=40, num_layers=3).cuda()
    decoder = Decoder(input_size=1, embd_size=10, hidden_size=40, num_layers=3).cuda()

    data = get_series(1461)
    train_dataset = time_series(data, input_window=50, output_window=5, type='train')
    test_dataset = time_series(data, input_window=50, output_window=5, type='test')
    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

    global_step = 0
    # train
    for epoch in range(1000):
        count = 0
        total_loss = 0
        encoder.train()
        decoder.train()
        for input_window, output_window in train_dataloader:
            input_window = input_window.permute(1, 0, 2).cuda()
            output_window = output_window.permute(1, 0, 2).cuda()
            decoder_input = output_window[0:-1]
            label = output_window[1:]
            decoder_len = label.shape[0]

            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            batch_size = input_window.shape[1]


            window_size = input_window.shape[0]
            loss = 0

            _, encoder_hidden = encoder(input_window)

            loss = 0
            decoder_hidden = encoder_hidden
            for di in range(decoder_len):
                mu, sigma, decoder_hidden = decoder(decoder_input[di:di+1], decoder_hidden)
                loss += loss_fn(mu, sigma, label)

            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            total_loss += loss.item()

            count += 1
            global_step += 1

            if count % 10 == 0:
                total_loss /= count
                count = 0
                print('[training] epoch[{}/{}], iter[{}], loss[{}]'.format(epoch, 1000, global_step, total_loss))
                total_loss = 0

        # test
        if epoch % 2 == 0:
            test(encoder, decoder, test_dataloader, epoch)


def test(encoder, decoder, dataloader,epoch):
    encoder.eval()
    decoder.eval()
    # 存放预测结果
    forecast = []

    for input_window, output_window in dataloader:
        input_window.cuda()
        output_window.cuda()



        _, encoder_hidden = encoder(input_window)





    plot_pred_and_real(results, real, epoch)



if __name__ == "__main__":
    train()
