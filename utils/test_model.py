import torch
from utils.plot_pred_real import plot_pred_and_real


def test_NBeats(net, epoch, vali_dataloader):
    net.eval()

    forecast = []
    real = []

    for input, target in vali_dataloader:
        back, fore = net(input)
        forecast.append(fore)
        real.append(target)

    forecast = torch.cat(forecast, dim=1).cpu().detach().numpy().reshape(-1)
    real = torch.cat(real, dim=0).cpu().detach().numpy().reshape(-1)
    vali_loss = plot_pred_and_real(forecast, real, epoch)
    return vali_loss


def test_RNNS(net, epoch, vali_dataloader):
    net.eval()

    forecast = []
    real = []

    for input, target in vali_dataloader:
        input = input.cuda()
        fore = net(input)
        forecast.append(fore)
        real.append(target)

    forecast = torch.cat(forecast, dim=1).cpu().detach().numpy().reshape(-1)
    real = torch.cat(real, dim=0).cpu().detach().numpy().reshape(-1)
    vali_loss = plot_pred_and_real(forecast, real, epoch)
    return vali_loss


def test_Transformer(net, epoch, vali_dataloader):
    net.eval()

    forecast = []
    real = []

    for input, target in vali_dataloader:
        input = input.cuda()
        fore = net(input)
        forecast.append(fore)
        real.append(target)

    forecast = torch.cat(forecast, dim=1).cpu().detach().numpy().reshape(-1)
    real = torch.cat(real, dim=0).cpu().detach().numpy().reshape(-1)
    vali_loss = plot_pred_and_real(forecast, real, epoch)
    return vali_loss