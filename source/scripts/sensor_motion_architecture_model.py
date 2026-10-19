"""Matched information masks and seven-output supervision/decoder controls."""
import torch
from torch import nn
from torch.nn import functional as F
from q3_sensor.endpoint_models import EndpointFusion, tokens
from q3_sensor.protocol import MODALITIES
from scripts.sensor_motion_pair_model import coarse_targets


def context(base, batch, mask='Full'):
    recurrent = torch.cat([base.encoders[m](batch[m]) for m in MODALITIES], -1)
    current = torch.cat([tokens(batch[m])[:, -1] for m in MODALITIES], -1)
    if mask == 'RecurrentOnly': current = current * 0
    elif mask == 'CurrentOnly': recurrent = recurrent * 0
    elif mask not in ['Full', 'FullUniform', 'FullBalanced']: raise ValueError(mask)
    return torch.cat((recurrent, current), -1)


def base_outputs(base, batch, mask='Full'):
    x = context(base, batch, mask)
    available = torch.stack([batch[m]['valid'].flatten(1).any(1) for m in MODALITIES], 1).any(1)
    b = torch.where(available[:, None], base.head(x), base.prior_logits[None])
    return x, b


def coarse(logits):
    return torch.stack((logits[:, 0], logits[:, 1], torch.logsumexp(logits[:, 2:], -1)), -1)


def pair(base_prediction, proposed):
    return torch.where((base_prediction <= 2) & (proposed <= 2), proposed, base_prediction)


class MaskedBase(EndpointFusion):
    def __init__(self, prior, initial_state, mask):
        super().__init__('gru', prior)
        self.load_state_dict(initial_state, strict=True)
        self.mask = mask

    def forward(self, batch):
        return base_outputs(self, batch, self.mask)[1]


class ArchitectureRefiner(nn.Module):
    def __init__(self, base, kind, branch_seed, mask='Full'):
        super().__init__()
        self.base = base.requires_grad_(False).eval()
        self.kind, self.mask = kind, mask
        self.outputs = 7 if kind in ['R7Fine', 'R7Coarse'] else 3
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(branch_seed)
            self.residual = nn.Sequential(nn.Linear(227, 24), nn.GELU(), nn.Linear(24, self.outputs))
            nn.init.zeros_(self.residual[-1].weight)
            nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, batch, motion):
        with torch.no_grad(): x, b = base_outputs(self.base, batch, self.mask)
        residual = self.residual(torch.cat((x, motion), -1))
        if self.outputs == 7:
            z = b + residual
            aux = coarse(z)
        else:
            z = None
            aux = coarse(b) + residual
        if not torch.isfinite(b).all() or not torch.isfinite(aux).all(): raise RuntimeError('Nonfinite refinement')
        p = b.argmax(-1) + 1
        predictions = {'pair3': pair(p, aux.argmax(-1) + 1)}
        if z is not None:
            if not torch.isfinite(z).all(): raise RuntimeError('Nonfinite seven logits')
            predictions.update(global7=z.argmax(-1)+1, pair7=pair(p, z.argmax(-1)+1))
        return dict(base_logits=b, auxiliary_logits=aux, logits7=z, predictions=predictions)


def weighted_loss(logits, y7, weights, grouped=False):
    target = coarse_targets(y7) if grouped else y7
    row = weights[y7]
    return (F.cross_entropy(logits, target, reduction='none') * row).sum() / row.sum()
