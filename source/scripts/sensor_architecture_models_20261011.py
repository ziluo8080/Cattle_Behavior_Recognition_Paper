"""Matched endpoint interfaces for LSTM, attention and a serial hybrid."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from q3_sensor.endpoint_models import EndpointFusion, tokens
from q3_sensor.protocol import MODALITIES


class SequenceEncoder(nn.Module):
    def __init__(self, features, kind):
        super().__init__()
        self.kind = kind
        self.project = nn.Linear(features * 2 + 2, 32)
        if kind == 'LSTM':
            self.lstm = nn.LSTM(32, 48, batch_first=True)
            self.output = nn.Linear(48, 64)
        else:
            count = 2 if kind == 'Transformer' else 1
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(32, 4, 64, dropout=.1, activation='gelu',
                                           batch_first=True, norm_first=True) for _ in range(count)])
            if kind == 'Transformer-LSTM':
                self.lstm = nn.LSTM(32, 32, batch_first=True)
            self.output = nn.Linear(32, 64)
            position = torch.arange(60, dtype=torch.float32)[:, None]
            frequency = torch.exp(torch.arange(0, 32, 2, dtype=torch.float32) * (-math.log(10000) / 32))
            encoding = torch.zeros(60, 32)
            encoding[:, 0::2] = torch.sin(position * frequency)
            encoding[:, 1::2] = torch.cos(position * frequency)
            self.register_buffer('position', encoding, persistent=True)
            self.register_buffer('causal_mask', torch.ones(60, 60, dtype=torch.bool).triu(1), persistent=False)
        self.norm = nn.LayerNorm(64)

    def sequence(self, data):
        value = F.gelu(self.project(tokens(data)))
        if self.kind != 'LSTM':
            value = value + self.position[None, :value.shape[1]]
            usable = data['valid'].any(-1).clone()
            # A missing sentinel prevents empty causal-prefix attention without claiming an observation.
            usable[:, 0] = True
            for layer in self.layers:
                value = layer(value, src_mask=self.causal_mask[:value.shape[1], :value.shape[1]],
                              src_key_padding_mask=~usable, is_causal=True)
        if self.kind != 'Transformer':
            value, _ = self.lstm(value)
        return self.norm(self.output(value))

    def forward(self, data):
        return self.sequence(data)[:, -1] * data['valid'][:, -1].any(-1, keepdim=True)


class AdaptedFusion(nn.Module):
    def __init__(self, kind, prior):
        super().__init__()
        self.encoders = nn.ModuleDict({m: SequenceEncoder(d, kind) for m, d in zip(MODALITIES, (3, 6, 1))})
        self.head = nn.Sequential(nn.Linear(218, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 7))
        self.register_buffer('prior_logits', torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8).log())

    def forward(self, batch):
        history = [self.encoders[m](batch[m]) for m in MODALITIES]
        current = [tokens(batch[m])[:, -1] for m in MODALITIES]
        logits = self.head(torch.cat(history + current, -1))
        available = torch.stack([batch[m]['valid'].flatten(1).any(1) for m in MODALITIES], 1).any(1)
        return torch.where(available[:, None], logits, self.prior_logits[None])


def make_model(kind, prior):
    return EndpointFusion('gru', prior) if kind == 'GRU' else AdaptedFusion(kind, prior)
