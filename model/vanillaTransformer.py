import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class TokenEmbedding(nn.Module):
    def __init__(self, input_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Linear(input_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens) * math.sqrt(self.emb_size)


class time_Transformer(nn.Module):
    def __init__(self, input_size=1, feature_size=256, num_layers=3, dropout=0.1, pred_len=10, device=None):
        super(time_Transformer, self).__init__()
        self.pred_len = pred_len
        self.feature_size = feature_size
        self.device = device

        self.input_mask = None
        self.TokenEmbedding = TokenEmbedding(input_size=input_size, emb_size=feature_size)
        self.pos_encoder = PositionalEncoding(feature_size, dropout=0.1)

        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_size, nhead=8, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_transformation = nn.Linear(feature_size, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.output_transformation.bias.data.zero_()
        self.output_transformation.weight.data.uniform_(-initrange, initrange)

    def forward(self, x):
        emb = self.TokenEmbedding(x)
        x = self.pos_encoder(emb)

        if self.input_mask is None or x.shape[1] != len(self.input_mask):
            self.input_mask = self._generate_square_subsequent_mask(x.shape[1]).to(self.device)

        output = self.encoder(src=x, mask=self.input_mask)

        output = output[:, -self.pred_len:, :]

        output = self.output_transformation(output)

        return output

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


if __name__ == "__main__":
    net = time_Transformer()
    input = torch.randn(32, 15, 1)
    output = net(input)
