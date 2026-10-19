"""Ordinary concatenation controls C0/C1, with a common all-missing fallback."""
import torch
from torch import nn
from torch.nn import functional as F

from .protocol import MODALITIES


class CausalConv(nn.Module):
    def __init__(self, channels_in, channels_out, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.conv = nn.Conv1d(channels_in, channels_out, 3, dilation=dilation)
        self.norm = nn.LayerNorm(channels_out)

    def forward(self, x):
        padded = F.pad(x.transpose(1, 2), (self.padding, 0))
        dilation = self.conv.dilation[0]
        if dilation > 1:
            # Deterministic torch2.4 CUDA otherwise uses slow_conv_dilated2d.
            # The order C,K matches Conv1d's cross-correlation weight layout.
            patches = padded.unfold(2, self.padding + 1, 1)[..., ::dilation]
            patches = patches.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[1], -1)
            out = F.linear(patches, self.conv.weight.flatten(1), self.conv.bias)
        else:
            out = self.conv(padded).transpose(1, 2)
        return F.gelu(self.norm(out))


class SensorEncoder(nn.Module):
    def __init__(self, features, include_age):
        super().__init__()
        self.include_age = include_age
        self.project = nn.Linear(features * 2 + 1 + int(include_age), 32)
        self.short = CausalConv(32, 64, 1)
        self.long = CausalConv(64, 64, 2)

    def forward(self, data):
        channels = [data['x'], data['valid'].float(), data['fresh'].float()]
        if self.include_age:
            age = data['age']
            normalized = torch.log1p(torch.nan_to_num(age, nan=60.0).clamp(0, 60)) / 4.110873864173311
            channels.append(normalized)
        features = self.long(self.short(F.gelu(self.project(torch.cat(channels, dim=-1)))))
        valid = data['valid'].any(dim=-1, keepdim=True)
        return (features * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)


class SensorConcat(nn.Module):
    def __init__(self, include_age, prior):
        super().__init__()
        self.encoders = nn.ModuleDict({m: SensorEncoder(d, include_age)
                                      for m, d in zip(MODALITIES, (3, 6, 1))})
        self.head = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, 7))
        self.register_buffer('prior_logits', torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8).log())

    def forward(self, batch):
        representations = [self.encoders[m](batch[m]) for m in MODALITIES]
        logits = self.head(torch.cat(representations, dim=1))
        available = torch.stack([batch[m]['valid'].flatten(1).any(dim=1) for m in MODALITIES], dim=1).any(dim=1)
        return torch.where(available[:, None], logits, self.prior_logits[None, :])
