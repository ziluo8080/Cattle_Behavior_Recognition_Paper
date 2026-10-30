"""Two-rate masked ConvTran with bidirectional feature interaction.

Adapted from the author's ConvTran convolution/tAPE/eRPE equations, with explicit
missing-key handling. This is not an exact reproduction of the old classifier.
Original copyright (c) 2022 Department of Data Science and Artificial Intelligence
@Monash University, MIT; license retained in third_party/convtran_context_20261022_v1/LICENSE.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean(sequence, valid, start=0):
    mask = valid[:, start:, None].bool()
    values = torch.where(mask, sequence[:, start:], 0.)
    return values.sum(1) / mask.sum(1).clamp_min(1)


def safe_attention(scores, key_valid):
    """Finite softmax for empty rows; absent keys receive exactly zero mass."""
    mask = key_valid[:, None, None, :].bool()
    weights = scores.masked_fill(~mask, -1e9).softmax(-1)
    return torch.where(mask, weights, 0.)


def time_encoding(length, width, t_ape=True):
    position = torch.arange(length, dtype=torch.float32)[:, None]
    frequency = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.) / width))
    angles = position * frequency * (width / length if t_ape else 1.)
    encoding = torch.zeros(length, width)
    encoding[:, 0::2], encoding[:, 1::2] = angles.sin(), angles.cos()
    return encoding[None]


class MaskedRelativeAttention(nn.Module):
    """Author's eRPE: add relative bias AFTER softmax, mask it before values."""
    def __init__(self, width, heads, length):
        super().__init__()
        self.heads, self.scale = heads, width ** -.5
        self.key = nn.Linear(width, width, bias=False)
        self.value = nn.Linear(width, width, bias=False)
        self.query = nn.Linear(width, width, bias=False)
        self.relative_bias_table = nn.Parameter(torch.zeros(2 * length - 1, heads))
        positions = torch.arange(length)
        self.register_buffer('relative_index', positions[:, None] - positions[None] + length - 1)
        self.to_out = nn.LayerNorm(width)

    def forward(self, sequence, valid):
        batch, length, width = sequence.shape
        def heads(layer):
            return layer(sequence).view(batch, length, self.heads, -1).transpose(1, 2)
        query, key, value = heads(self.query), heads(self.key), heads(self.value)
        weights = safe_attention((query @ key.transpose(-1, -2)) * self.scale, valid)
        bias = self.relative_bias_table[self.relative_index].permute(2, 0, 1)[None]
        weights = torch.where(valid[:, None, None, :], weights + bias, 0.)
        mixed = (weights @ value).transpose(1, 2).reshape(batch, length, width)
        result = self.to_out(mixed)
        usable = valid & valid.any(1, keepdim=True)
        return torch.where(usable[:, :, None], result, 0.)


class MaskedConvTran(nn.Module):
    def __init__(self, channels, length, width=32, heads=4, ff=128, dropout=.1):
        super().__init__()
        self.embedding = nn.Sequential(nn.Conv2d(1, 4 * width, (1, 8), padding='same'),
            nn.BatchNorm2d(4 * width), nn.GELU(),
            nn.Conv2d(4 * width, width, (channels, 1)), nn.BatchNorm2d(width), nn.GELU())
        self.register_buffer('position', time_encoding(length, width))
        self.position_dropout = nn.Dropout(dropout)
        self.attention = MaskedRelativeAttention(width, heads, length)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.feedforward = nn.Sequential(nn.Linear(width, ff), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(ff, width), nn.Dropout(dropout))

    def forward(self, values, valid):
        values = torch.where(valid[:, :, None], values, 0.)
        embedded = self.embedding(values.transpose(1, 2)[:, None]).squeeze(2).transpose(1, 2)
        embedded = torch.where(valid[:, :, None], embedded, 0.)
        positioned = self.position_dropout(embedded + self.position)
        attended = self.norm1(embedded + self.attention(positioned, valid))
        output = self.norm2(attended + self.feedforward(attended))
        return torch.where(valid[:, :, None], output, 0.)


class CrossDirection(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.heads, self.scale = heads, (width // heads) ** -.5
        self.query, self.key, self.value = [nn.Linear(width, width) for _ in range(3)]
        self.output = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, queries, keys, query_valid, key_valid):
        batch, q_length, width = queries.shape
        k_length = keys.shape[1]
        q = self.query(queries).view(batch, q_length, self.heads, -1).transpose(1, 2)
        k = self.key(keys).view(batch, k_length, self.heads, -1).transpose(1, 2)
        v = self.value(keys).view(batch, k_length, self.heads, -1).transpose(1, 2)
        weights = self.dropout(safe_attention((q @ k.transpose(-1, -2)) * self.scale, key_valid))
        delta = (weights @ v).transpose(1, 2).reshape(batch, q_length, width)
        delta = self.output(delta)
        support = query_valid & key_valid.any(1, keepdim=True)
        combined = self.norm(queries + self.dropout(delta))
        # Empty other branch preserves a valid query's encoded representation.
        result = torch.where(support[:, :, None], combined, queries)
        return torch.where(query_valid[:, :, None], result, 0.)


class TwoRateConvTran(nn.Module):
    def __init__(self, cfg, prior, interaction=True):
        super().__init__()
        width = cfg['width']
        options = dict(width=width, heads=cfg['heads'], ff=cfg['ff'], dropout=cfg['dropout'])
        self.dynamic = MaskedConvTran(8, 100, **options)
        self.state = MaskedConvTran(26, 60, **options)
        self.context = nn.Sequential(nn.Linear(136, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(4 * width + 64 + 26, 128), nn.GELU(),
                                  nn.Dropout(cfg['dropout']), nn.Linear(128, 7))
        # Common modules are initialized before optional interaction modules.
        self.interaction = interaction
        if interaction:
            self.dynamic_from_state = CrossDirection(width, cfg['heads'], cfg['dropout'])
            self.state_from_dynamic = CrossDirection(width, cfg['heads'], cfg['dropout'])
        self.register_buffer('prior_logits', torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8).log())

    def forward(self, batch):
        dynamic_valid = batch['dynamic'][:, :, 6] > .5
        physical = torch.where(dynamic_valid[:, :, None], batch['dynamic'][:, :, :6], 0.)
        age = torch.where(dynamic_valid[:, :, None], batch['dynamic'][:, :, 7:].clamp(0, .2), .2)
        cleaned = torch.cat((physical, dynamic_valid[:, :, None].float(), age), -1)
        dynamic = self.dynamic(cleaned, dynamic_valid)
        state_valid = batch['state_valid'].bool()
        state = self.state(batch['state'], state_valid)
        if self.interaction:
            dynamic, state = (self.dynamic_from_state(dynamic, state, dynamic_valid, state_valid),
                              self.state_from_dynamic(state, dynamic, state_valid, dynamic_valid))
        readouts = [masked_mean(dynamic, dynamic_valid), masked_mean(dynamic, dynamic_valid, 90),
                    masked_mean(state, state_valid), masked_mean(state, state_valid, 55),
                    self.context(batch['context']), batch['current']]
        logits = self.head(torch.cat(readouts, -1))
        available = dynamic_valid.any(1) | state_valid.any(1)
        return torch.where(available[:, None], logits, self.prior_logits[None])


def make_model(arm, cfg, prior):
    if arm == 'Original_ConvTran':
        # Original class and original cloud vendor are kept untouched.
        from scripts.sensor_native_convtran_20261021 import NativeConvTran
        return NativeConvTran()
    if arm in ('TwoRate_Concat', 'TwoRate_Interaction'):
        return TwoRateConvTran(cfg, prior, interaction=arm == 'TwoRate_Interaction')
    raise ValueError('Unregistered architecture')
