import torch
import torch.utils
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from typing import *
from utils.plot_pred_real import plot_pred_and_real

from sklearn.preprocessing import MinMaxScaler, StandardScaler


class futures(Dataset):
    def __init__(self, root: str = "../Corn.csv", target_col: str = "volume",
                 input_len: int = 50, output_len: int = 5, step: int = -1,
                 data_type: str = "train", split_rate: float = 0.9, is_scale=True):
        self.input_len = input_len
        self.output_len = output_len
        self.is_scale = is_scale
        self.raw_data = self.read_data(root, target_col)
        train_len = int(len(self.raw_data) * split_rate)
        test_len = vali_len = int((1 - split_rate) * len(self.raw_data) * 0.5)

        # 初始化一个scaler
        self.scaler = StandardScaler()

        if data_type == "train":
            self.data = self.raw_data[:train_len]
            self.step = 1 if step == -1 else step

        if data_type == "test":
            self.data = self.raw_data[train_len:train_len+test_len]
            self.step = self.output_len if step == -1 else step

        if data_type == "vali":
            self.data = self.raw_data[train_len+vali_len:train_len+test_len+vali_len]
            self.step = self.output_len if step == -1 else step

    def read_data(self, root: str, target_col: str):
        df = pd.read_csv(root)
        data = df[target_col].values
        return data

    def __getitem__(self, index):
        start_index = self.step * index
        end_index = start_index + self.input_len + self.output_len
        window = self.data[start_index: end_index]
        input_data = window[:self.input_len]
        output_data = window[self.input_len:]
        if self.is_scale:
            input_data = self.scaler.fit_transform(input_data.reshape(-1, 1)).reshape(-1)
            output_data = self.scaler.fit_transform(output_data.reshape(-1, 1)).reshape(-1)
        input_data = torch.tensor(input_data, dtype=torch.float)
        output_data = torch.tensor(output_data, dtype=torch.float)
        return input_data.float(), output_data.float()

    def __len__(self):
        return (len(self.data) - self.output_len - self.input_len) // self.step


if __name__ == '__main__':
    dataset = features(data_type='test')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1)

    outputs = []
    for input_data, output_data in dataloader:
        outputs.append(output_data)
    outputs = torch.cat(outputs, dim=1).cpu().detach().numpy().reshape(-1)
    vali_loss = plot_pred_and_real(outputs, outputs, 1)
