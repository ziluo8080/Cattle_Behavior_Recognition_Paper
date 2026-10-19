"""Same causal TCN parameters; channels-last 2D kernels avoid the slow dilated 1D backend."""
import torch
from torch import nn
from torch.nn import functional as F
from q3_sensor.endpoint_models import tokens
from q3_sensor.protocol import MODALITIES


class FastConv1d(nn.Conv1d):
    def forward(self, value):
        # A height-one 2D convolution is the same 1D operation. Keep 3D
        # parameter tensors and initialization so checkpoint keys do not change.
        value = value.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        weight = self.weight.unsqueeze(2).contiguous(memory_format=torch.channels_last)
        return F.conv2d(value, weight, self.bias, stride=(1, self.stride[0]),
            padding=(0, self.padding[0]), dilation=(1, self.dilation[0]),
            groups=self.groups).squeeze(2)


class ResidualBlock(nn.Module):
    def __init__(self, width, dilation):
        super().__init__()
        self.padding = 2 * dilation
        self.convs = nn.ModuleList([FastConv1d(width, width, 3, dilation=dilation) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.drop = nn.Dropout(.1)

    def forward(self, value):
        original = value
        for conv, norm in zip(self.convs, self.norms):
            value = conv(F.pad(value.transpose(1, 2), (self.padding, 0))).transpose(1, 2)
            value = self.drop(F.gelu(norm(value)))
        return value + original


class TCNEncoder(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.project = nn.Linear(features * 2 + 2, 32)
        self.input = nn.Linear(32, 23)
        self.blocks = nn.ModuleList([ResidualBlock(23, d) for d in (1, 2, 4, 8, 16)])
        self.output = nn.Linear(23, 64)
        self.norm = nn.LayerNorm(64)

    def sequence(self, data):
        value = F.gelu(self.input(F.gelu(self.project(tokens(data)))))
        for block in self.blocks:
            value = block(value)
        return self.norm(self.output(value))

    def forward(self, data):
        return self.sequence(data)[:, -1] * data['valid'][:, -1].any(-1, keepdim=True)


class TCNFusion(nn.Module):
    def __init__(self, prior):
        super().__init__()
        self.encoders = nn.ModuleDict({m: TCNEncoder(d) for m, d in zip(MODALITIES, (3, 6, 1))})
        self.head = nn.Sequential(nn.Linear(218, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 7))
        self.register_buffer('prior_logits', torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8).log())

    def forward(self, batch):
        history = [self.encoders[m](batch[m]) for m in MODALITIES]
        current = [tokens(batch[m])[:, -1] for m in MODALITIES]
        logits = self.head(torch.cat(history + current, -1))
        available = torch.stack([batch[m]['valid'].flatten(1).any(1) for m in MODALITIES], 1).any(1)
        return torch.where(available[:, None], logits, self.prior_logits[None])
