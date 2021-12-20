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
from model.RNNS import Net


def train(model, input_len, output_len):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_dataset = time_series(input_len=input_len, pred_len=output_len, type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    vali_dataset = time_series(input_len=input_len, pred_len=output_len, type='test')
    vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)

    global_step = 0
    for epoch in range(1000):
        count = 0
        total_loss = 0
        net.train()
        train_bar = tqdm(train_dataloader)
        for x, y in train_bar:
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
                train_bar.desc = '[training] [epoch:{}/{} , iter:{}, total_loss:{}]'\
                    .format(epoch, 1000, global_step, total_loss)
                total_loss = 0

            optimizer.step()

        if epoch % 5 == 0:
            test(net, epoch, vali_dataloader)


if __name__ == "__main__":
    input_len, output_len = 5, 1
    net = Net(in_size=input_len, embd_size=40, out_size=output_len, layer_num=3).cuda()
    train(net, input_len, output_len)