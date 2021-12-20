from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm

from data.datasets.time_series import time_series
from data.datasets.features import features
from utils.test_model import test_MLP as test

from model.MLP import MLP as Net


def train(net, input_len, output_len):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    # train_dataset = time_series(input_len=input_len, pred_len=output_len, type='train')
    # train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    # vali_dataset = time_series(input_len=input_len, pred_len=output_len, type='test')
    # vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)
    train_dataset = features(root='./data/Corn.csv', input_len=input_len, output_len=output_len, data_type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    vali_dataset = features(root='./data/Corn.csv', input_len=input_len, output_len=output_len, data_type='vali')
    vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)

    net.train()
    global_step = 0
    for epoch in range(1000):
        count = 0
        total_loss = 0
        train_bar = tqdm(train_dataloader)
        for x, y in train_bar:
            x = x.cuda()
            y = y.cuda()
            optimizer.zero_grad()
            out = net(x)
            loss = loss_fn(out, y)
            loss.backward()

            count += 1
            global_step += 1
            total_loss += loss.item()
            if count % 10 == 0:
                total_loss /= count
                count = 0
                train_bar.desc = '[training] [epoch:{}/{} , iter:{}, total_loss:{}]'\
                    .format(epoch, 1000, global_step, total_loss)
            optimizer.step()

        if epoch % 10 == 0:
            test(net, epoch, vali_dataloader)


if __name__ == "__main__":
    input_len, output_len = 20, 10
    net = Net(in_len=input_len, out_len=output_len).cuda()
    train(net, input_len, output_len)

