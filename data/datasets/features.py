import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from typing import *


class features(Dataset):
    def __init__(self, root: str="../Corn.csv", target_col: str="volume",
                 input_len: int=50, output_len: int=5, step: int=-1,
                 data_type: str="train", split_rate: int=0.9):
        self.input_len = input_len
        self.output_len = output_len

        self.raw_data = self.read_data(root, target_col)
        train_len = int(len(self.raw_data) * split_rate)
        test_len = vali_len = int((1 - split_rate) * len(self.raw_data) * 0.5)
        if data_type == "train":
            self.data = torch.tensor(self.raw_data[:train_len])
            self.step = 1 if step == -1 else step

        if data_type == "test":
            self.data = torch.tensor(self.raw_data[train_len:train_len+test_len])
            self.step = 1 if step == -1 else step

        if data_type == "vali":
            self.data = torch.tensor(self.raw_data[train_len+vali_len:train_len+test_len+vali_len])
            self.step = 1 if step == -1 else step

    def read_data(self, root: str, target_col: str):
        df = pd.read_csv(root)
        data = df[target_col].values
        return data

    def __getitem__(self, index):
        input_data = self.data[index: index+self.input_len]
        output_data = self.data[index+self.input_len: index+self.input_len+self.output_len]

        return input_data, output_data

    def __len__(self):
        return (len(self.data) - self.output_len - self.input_len) // self.step


if __name__ == '__main__':
    dataset = features(data_type='test')
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=1)

    count = 0
    for input_data, output_data in dataloader:
        count += 1
    print(count)