import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class LatentMappingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LatentMappingLayer, self).__init__()
        self.num_layers = 2
        self.enc = nn.ModuleList([
            nn.Linear(input_dim, 512)
        ])
        for i in range(1, 2):
            if i == 2 - 1:
                self.enc.append(nn.Linear(512, output_dim))
            else:
                self.enc.append(nn.Linear(512, 512))

    def forward(self, x, dropout=0.1):
        z = self.encode(x, dropout)
        return z

    def encode(self, x, dropout=0.1):
        h = x
        for i, layer in enumerate(self.enc):
            if i == self.num_layers - 1:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
            else:
                if dropout:
                    h = torch.dropout(h, dropout, train=self.training)
                h = layer(h)
                h = F.tanh(h)
        return h

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, hidden_dim, attention_dropout_rate,
                 num_heads, attn_bias_dim, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_models = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim))

        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.attention_net = MultiHeadAttention(hidden_dim, attention_dropout_rate, num_heads,
                                           attn_bias_dim)

        self.view = view

    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        ls = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.feature_contrastive_models(z), dim=1)
            l = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            zs.append(z)
            hs.append(h)
            xrs.append(xr)
            ls.append(l)
        return xrs, zs, hs, ls

    def forward_cluster(self, xs):
        qs = []
        preds = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        return qs, preds

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, num_heads, attn_bias_dim):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads

        self.att_size = att_size = hidden_size // num_heads
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_k = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_v = nn.Linear(hidden_size, num_heads * att_size)
        self.linear_bias = nn.Linear(attn_bias_dim, num_heads)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(num_heads * att_size, 1)

    def forward(self, q, k, v):

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        q = self.linear_q(q).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2)  # [b, h, q_len, d_k]
        k = self.linear_k(k).view(batch_size, -1, self.num_heads, d_k).transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]
        v = self.linear_v(v).view(batch_size, -1, self.num_heads, d_v).transpose(1, 2)  # [b, h, v_len, d_v]

        q = q * self.scale
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]

        x = torch.softmax(x, dim=3)
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, self.num_heads * d_v)

        x = self.output_layer(x)

        return x