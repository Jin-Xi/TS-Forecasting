import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from utils.losses import smape_2_loss, mape_loss, mase_loss

from N_Beats import NBeatsNet
import numpy as np

from utils.plot_pred_real import plot_pred_and_real

"""
different for previous version, in this model we try a real seq2seq model
"""

torch.manual_seed(888)
logger = logging.getLogger('DeepAR.Net')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
工具函数
"""


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


"""
数据集定义
"""


class time_series(Dataset):
    def __init__(self, window_size=60, input_len=50, pred_len=10, type='train'):
        assert window_size == (input_len + pred_len)
        self.window_size = window_size
        self.pred_len = pred_len
        self.type = type
        self.input_len = input_len

        self.series, self.split_time, self.x_train, self.x_valid, self.time_train, self.time_valid = self.init_data()

        if type == 'train':
            self.data = self.x_train
            self.time = self.time_train
            self.step = 1
        if type == 'test':
            self.data = self.x_valid
            self.time = self.time_valid
            self.step = self.pred_len

    def init_data(self):
        time = np.arange(4 * 365 + 1, dtype="float32")
        baseline = 10
        series = trend(time, 0.1)
        baseline = 10
        amplitude = 40
        slope = 0.05
        noise_level = 5

        # Create the series
        series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
        # Update with noise
        series += noise(time, noise_level, seed=42)

        split_time = 1000
        time_train = time[:split_time]
        x_train = series[:split_time]
        time_valid = time[split_time:]
        x_valid = series[split_time:]
        return series, split_time, x_train, x_valid, time_train, time_valid

    def __getitem__(self, index):
        start = index * self.step
        end = start + self.window_size
        window = self.data[start:end]
        input_tensor = torch.tensor(window[:self.input_len, None])
        target_tensor = torch.tensor(window[self.input_len:, None])
        return torch.squeeze(input_tensor), torch.squeeze(target_tensor)

    def __len__(self):
        return (len(self.data) - self.window_size) // self.step


def train():
    model = NBeatsNet(backcast_length=40, forecast_length=10,
                      stack_types=(NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK, NBeatsNet.GENERIC_BLOCK,
                                   NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.TREND_BLOCK),
                      nb_blocks_per_stack=3,
                      thetas_dim=(4, 4, 4, 4, 4), share_weights_in_stack=True, hidden_layer_units=64)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    # losses_dict = {"MAPE": mape_loss, "MASE": mase_loss, "SMAPE": smape_2_loss}
    loss_fn = torch.nn.MSELoss()
    dataset = time_series(window_size=50, input_len=40, pred_len=10, type='train')
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    global_step = 0
    for epoch in range(1000):
        count = 0
        total_loss = 0
        model.train()
        train_bar = tqdm(dataloader)
        for input, target in train_bar:
            input = input.cuda()
            target = target.cuda()

            optimizer.zero_grad()

            backcast, forecast = model(input)

            loss = loss_fn(forecast, target)
            loss.backward()

            total_loss += loss.item()
            count += 1
            global_step += 1

            optimizer.step()

            if count % 100 == 0:
                total_loss /= count
                train_bar.desc = '[training] epoch[{}/{}], iter[{}], loss[{}]'.format(epoch, 1000, global_step,
                                                                                      total_loss)
                count = 0
                total_loss = 0

        if epoch % 5 == 0:
            vali_loss = test(model, epoch)
            scheduler.step(vali_loss)


def test(net, epoch):
    net.eval()

    forecast = []
    real = []

    dataset = time_series(window_size=50, input_len=40, pred_len=10, type='test')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    for input, target in dataloader:
        back, fore = net(input)
        forecast.append(fore)
        real.append(target)
    forecast = torch.cat(forecast, dim=1).cpu().detach().numpy().reshape(-1)
    real = torch.cat(real, dim=1).cpu().detach().numpy().reshape(-1)
    vali_loss = plot_pred_and_real(forecast, real, epoch)
    return vali_loss

if __name__ == "__main__":
    train()
    # dataset = time_series(window_size=50, input_len=40, pred_len=10, type='test')
    # print(len(dataset))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    # count = 0
    # for input, target in dataloader:
    #     count += 1
    # print(count)
