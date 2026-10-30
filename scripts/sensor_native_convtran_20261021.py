"""Separate same-information ConvTran control; all differentiation stays in Torch."""
import sys
from pathlib import Path

import torch
from torch import nn

VENDOR = Path(__file__).resolve().parents[1] / 'third_party/convtran_20261021_v1'
sys.path.insert(0, str(VENDOR))
from Models.model import ConvTran


class NativeConvTran(nn.Module):
    def __init__(self):
        super().__init__()
        config = dict(Data_shape=[None, 8, 100], emb_size=16, num_heads=8,
                      dim_ff=256, dropout=.1, Fix_pos_encode='tAPE', Rel_pos_encode='eRPE')
        self.encoder = ConvTran(config, num_classes=7)
        # Keep the official computation through LayerNorm2; expose N,C,T map.
        self.encoder.gap = nn.Identity()
        self.encoder.flatten = nn.Identity()
        self.encoder.out = nn.Identity()
        self.context = nn.Sequential(nn.Linear(136, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(16 * 2 + 64, 128), nn.ReLU(),
                                  nn.Dropout(.1), nn.Linear(128, 7))

    def encode(self, dynamic):
        valid = dynamic[:, :, 6] > .5
        physical = torch.where(valid[:, :, None], dynamic[:, :, :6], 0.)
        age = torch.where(valid[:, :, None], dynamic[:, :, 7:].clamp(0, .2), .2)
        cleaned = torch.cat([physical, valid[:, :, None].float(), age], dim=2)
        features = self.encoder(cleaned.transpose(1, 2)).transpose(1, 2)
        readouts = []
        for start in (0, 90):
            mask = valid[:, start:, None]
            total = torch.where(mask, features[:, start:], 0.).sum(1)
            count = mask.sum(1)
            readouts.append(total / count.clamp_min(1))
        return torch.cat(readouts, dim=1)

    def forward(self, batch):
        return self.head(torch.cat([self.encode(batch['dynamic']), self.context(batch['context'])], dim=1))
