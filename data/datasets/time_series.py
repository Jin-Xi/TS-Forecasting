from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
import torch
from torch import Tensor
import numpy as np

from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, input_len=50, pred_len=10, type='train'):
        self.window_size = input_len + pred_len
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


class time_series2(Dataset):
    def __init__(self, input_len=50, pred_len=10, type='train'):
        self.window_size = input_len + pred_len
        self.pred_len = pred_len
        self.type = type
        self.input_len = input_len
        self.scalar = MinMaxScaler(feature_range=(-1, 1))

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
        # series = self._data_trainform(series)

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

    def _data_trainform(self, data):
        return torch.tensor(self.scalar.fit_transform(data.reshape(-1, 1)).reshape(-1))

    def _data_reverse_transform(self, data: Tensor):
        return torch.tensor(self.scalar.inverse_transform(data.numpy().reshape(-1, 1)).reshape(-1))


if __name__ == "__main__":
    dataset = time_series2()