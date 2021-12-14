from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn

from tqdm import tqdm

from utils.plot_pred_real import plot_pred_and_real
from utils.test_model import test_RNNS as test
from data.datasets.time_series import time_series


torch.manual_seed(888)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(net, input_len, output_len):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    train_dataset = time_series(window_size=input_len+output_len, input_len=input_len, pred_len=output_len, type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    vali_dataset = time_series(window_size=input_len+output_len, input_len=input_len, pred_len=output_len, type='test')
    vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)
    global_step = 0
    # train
    for epoch in range(1000):
        count = 0
        total_loss = 0
        net.train()
        train_bar = tqdm(train_dataloader)
        for input, target in train_bar:
            # TODO: 完成DeepAR训练过程
            pass


            loss = loss_fn(decoder_output, target)
            loss.backward()

            total_loss += loss.item()
            count += 1
            global_step += 1

            # grad_clip
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.1)
            optimizer.step()

            if count % 10 == 0:
                total_loss /= count
                train_bar.desc = '[training] epoch[{}/{}], iter[{}], loss[{}]'.format(epoch, 1000, global_step, total_loss)
                count = 0
                total_loss = 0

        # test
        if epoch % 2 == 0:
            vali_loss = test(net, epoch, vali_dataloader)


if __name__ == "__main__":
    # TODO: 测试train模块
    pass

