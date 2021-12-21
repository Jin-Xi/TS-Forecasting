from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

import logging
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler

from utils.test_model import test_TransformerEncoder as test
from data.datasets.time_series import time_series as time_series
from data.datasets.futures import futures
from model.Transformer import time_TransformerEncoder as Net

"""
different for previous version, in this model we try a real seq2seq model
"""

torch.manual_seed(888)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(net, input_len, output_len):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # train_dataset = time_series(input_len=input_len, pred_len=output_len, type='train')
    # train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    # vali_dataset = time_series(input_len=input_len, pred_len=output_len, type='test')
    # vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)
    train_dataset = futures(root='./data/Corn.csv', input_len=input_len, output_len=output_len,
                            split_rate=0.8, data_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    vali_dataset = futures(root='./data/Corn.csv', input_len=input_len, output_len=output_len,
                           split_rate=0.8, data_type='vali')
    vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)

    global_step = 0

    # train
    for epoch in range(1000):
        count = 0
        total_loss = 0
        net.train()
        train_bar = tqdm(train_dataloader)
        for input, target in train_bar:
            input = input.unsqueeze(-1).cuda()
            target = torch.cat([input, target.cuda().unsqueeze(-1)], dim=1)[:, output_len:]
            optimizer.zero_grad()

            output = net(input)

            loss = loss_fn(output, target)
            loss.backward()

            total_loss += loss.item()
            count += 1
            global_step += 1


            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.7)
            optimizer.step()

            if count % 10 == 0:
                total_loss /= count
                train_bar.desc = '[training] epoch[{}/{}], iter[{}], loss[{}]'.format(epoch, 1000, global_step,
                                                                                      total_loss)
                count = 0
                total_loss = 0

        # test
        if epoch % 10 == 0:
            vali_loss = test(net, epoch, vali_dataloader, pred_len=output_len)


if __name__ == "__main__":
    input_len, output_len = 50, 5
    model = Net(feature_size=256, num_layers=1, dropout=0.1, pred_len=output_len, device=device).cuda()
    train(model, input_len, output_len)
    # 使用3层的transformer encoder输出就是一条直线
    # 使用2层的transformer encoder输出
    # 使用1层的transformer encoder 输出就和LSTM大差不差
    # 如果input_len太小 输出近似为直线
