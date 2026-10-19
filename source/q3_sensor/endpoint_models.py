"""Endpoint readouts with the same sensor values/masks/ages as C1."""
import torch
from torch import nn
from torch.nn import functional as F

from .model import SensorEncoder
from .protocol import MODALITIES


def tokens(data):
    age = torch.log1p(torch.nan_to_num(data['age'], nan=60.0).clamp(0, 60)) / 4.110873864173311
    return torch.cat((data['x'], data['valid'].float(), data['fresh'].float(), age), dim=-1)


class EndpointTCN(SensorEncoder):
    def __init__(self, features):
        super().__init__(features, include_age=True)

    def forward(self, data):
        out = self.long(self.short(F.gelu(self.project(tokens(data)))))
        return out[:, -1] * data['valid'][:, -1].any(-1, keepdim=True)


class EndpointGRU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.project = nn.Linear(features * 2 + 2, 32)
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.norm = nn.LayerNorm(64)

    def forward(self, data):
        out, _ = self.gru(F.gelu(self.project(tokens(data))))
        return self.norm(out[:, -1]) * data['valid'][:, -1].any(-1, keepdim=True)


class EndpointFusion(nn.Module):
    def __init__(self, kind, prior):
        super().__init__()
        encoder = {'tcn': EndpointTCN, 'gru': EndpointGRU}[kind]
        self.encoders = nn.ModuleDict({m: encoder(d) for m, d in zip(MODALITIES, (3, 6, 1))})
        # Raw endpoint skip preserves absolute location/posture information.
        self.head = nn.Sequential(nn.Linear(192 + 26, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 7))
        self.register_buffer('prior_logits', torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8).log())

    def forward(self, batch):
        z = [self.encoders[m](batch[m]) for m in MODALITIES]
        latest = [tokens(batch[m])[:, -1] for m in MODALITIES]
        logits = self.head(torch.cat(z + latest, dim=-1))
        available = torch.stack([batch[m]['valid'].flatten(1).any(1) for m in MODALITIES], 1).any(1)
        return torch.where(available[:, None], logits, self.prior_logits[None])
