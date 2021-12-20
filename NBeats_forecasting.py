import logging
from tqdm import tqdm

from torch.utils.data import DataLoader
import torch
from torch import optim

# from utils.losses import smape_2_loss, mape_loss, mase_loss
from model.N_Beats import NBeatsNet
from data.datasets.time_series import time_series

from utils.test_model import test_NBeats as test

"""
train_N-Beats!
"""

torch.manual_seed(888)
logger = logging.getLogger('DeepAR.Net')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, input_len, output_len):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)
    # losses_dict = {"MAPE": mape_loss, "MASE": mase_loss, "SMAPE": smape_2_loss}
    loss_fn = torch.nn.MSELoss()

    train_dataset = time_series(input_len=input_len, pred_len=output_len, type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    vali_dataset = time_series(input_len=input_len, pred_len=output_len, type='test')
    vali_dataloader = DataLoader(vali_dataset, batch_size=1, shuffle=False)



    global_step = 0
    for epoch in range(10000):
        count = 0
        total_loss = 0
        model.train()
        train_bar = tqdm(train_dataloader)
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
            vali_loss = test(model, epoch, vali_dataloader)
            scheduler.step(vali_loss)


if __name__ == "__main__":
    input_len = 20
    output_len = 40
    model = NBeatsNet(backcast_length=input_len, forecast_length=output_len,
                      stack_types=(NBeatsNet.SEASONALITY_BLOCK, NBeatsNet.TREND_BLOCK, NBeatsNet.GENERIC_BLOCK),
                      nb_blocks_per_stack=3,
                      thetas_dim=(4, 4, 4), share_weights_in_stack=False, hidden_layer_units=64)
    train(model, input_len, output_len)