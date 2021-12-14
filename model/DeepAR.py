from random import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#####################################################
#
#   定义网络：
#           应该使用encoder 和 decoder 结构
#
#####################################################


class EncoderRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.layerNorm = nn.LayerNorm(hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output = self.layerNorm(output)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size=10, output_size=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.layerNorm = nn.LayerNorm(hidden_size)
        # self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output = self.layerNorm(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class Net(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=10):
        super(Net, self).__init__()
        self.encoder = EncoderRNN(input_size=1, hidden_size=40)
        self.decoder = DecoderRNN(hidden_size=40, output_size=1)

    def forward(self, input_tensor, target_tensor):
        batch_size = input_tensor.size(0)
        input_size = input_tensor.size(2)
        encoder_hidden = self.encoder.initHidden(batch_size)

        input_length = input_tensor.size(1)
        target_length = target_tensor.size(1)

        encoder_outputs = torch.zeros(batch_size, input_length, self.encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[:, ei, None], encoder_hidden)
            encoder_outputs[:, ei:ei + 1, :] = encoder_output

        decoder_input = input_tensor[:, -1, None]

        decoder_hidden = encoder_hidden

        loss = 0
        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random() < teacher_forcing_ratio else False
        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                # loss += criterion(decoder_output, target_tensor[di])
                decoder_outputs.append(decoder_output)
                decoder_input = target_tensor[:, di:di + 1, :]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs

    def init_hidden(self, batch_size):
        return self.encoder.initHidden(batch_size)

    @staticmethod
    def loss_fn(mu: Variable, sigma: Variable, labels: Variable):
        distribution = torch.distributions.normal.Normal(torch.squeeze(mu), torch.squeeze(sigma))
        likelihood = distribution.log_prob(torch.squeeze(labels))
        return -torch.mean(likelihood)

if __name__ == '__main__':
    # TODO: 测试基础模型
    pass
