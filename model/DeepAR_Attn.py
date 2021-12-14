import torch
import torch.nn as nn
import torch.nn.functional as F

import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EncoderRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, num_layers=1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, num_layers=num_layers)
        self.norm = nn.InstanceNorm1d(hidden_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output = embedded
        output = self.norm(output)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size, device=device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size=40, output_size=1, input_len=50, dropout_p=0.1, num_layers=1):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.input_len = input_len

        self.use_layer_combine = True if num_layers > 1 else False

        self.embedding = nn.Linear(self.output_size, self.hidden_size)
        self.layer_combine = nn.Linear(num_layers * self.hidden_size, self.hidden_size)

        self.attn = nn.Linear(self.hidden_size * 2, self.input_len)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=num_layers, batch_first=True)

        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.norm = nn.InstanceNorm1d(hidden_size)

    def forward(self, x, hidden, encoder_outputs):
        raw_hidden = hidden
        embedded = self.embedding(x)
        embedded = self.norm(embedded)
        embedded = self.dropout(embedded)

        # 当有多层网络的时候融合
        if self.use_layer_combine:
            hidden = hidden.permute(1, 0, 2)
            hidden = self.layer_combine(hidden)
        else:
            hidden = hidden.permute(1, 0, 2)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden), 2)), dim=2)
        attn_applied = torch.bmm(attn_weights,
                                 encoder_outputs)

        output = torch.cat((embedded, attn_applied), dim=2)
        output = self.attn_combine(output)

        output = F.relu(output)

        output, hidden = self.gru(output, raw_hidden)

        output = self.out(output)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class DeepAR_Attn(nn.Module):
    def __init__(self, input_size=1, output_size=1, hidden_size=128, input_len=50, pred_len=10):
        super(DeepAR_Attn, self).__init__()
        self.encoder = EncoderRNN(input_size=input_size, hidden_size=hidden_size)
        self.decoder = AttnDecoderRNN(hidden_size=hidden_size, output_size=output_size,
                                      input_len=input_len)
        self.input_len = input_len
        self.pred_len = pred_len
        self.Attn = None

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

        decoder_input = input_tensor[:, -1, :].unsqueeze(1)

        decoder_hidden = encoder_hidden

        teacher_forcing_ratio = 0.5
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                # loss += criterion(decoder_output, target_tensor[di])
                decoder_outputs.append(decoder_output)
                decoder_input = target_tensor[:, di:di + 1, :]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, attn_weights = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_outputs.append(decoder_output)

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        return decoder_outputs

    def init_hidden(self, batch_size):
        return self.encoder.initHidden(batch_size)


if __name__ == '__main__':
    # TODO: 测试基础模型
    pass