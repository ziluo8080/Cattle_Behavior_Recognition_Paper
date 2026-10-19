"""Frozen successful mainline plus a fixed walking/standing refinement interface."""
import torch
from torch import nn
from q3_sensor.endpoint_models import EndpointFusion, tokens
from q3_sensor.protocol import MODALITIES


class MotionPairRefiner(nn.Module):
    def __init__(self, kind, prior, trained_base_state, branch_seed):
        super().__init__()
        self.kind = kind
        self.base = EndpointFusion('gru', prior)
        self.base.load_state_dict(trained_base_state, strict=True)
        self.base.requires_grad_(False).eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(branch_seed)
            if kind == 'BiasPair':
                self.bias = nn.Parameter(torch.zeros(3))
            elif kind == 'ContextPair':
                self.residual = nn.Sequential(nn.Linear(218, 25), nn.GELU(), nn.Linear(25, 3))
            elif kind in ['NoMotionPair', 'MotionPair']:
                self.residual = nn.Sequential(nn.Linear(227, 24), nn.GELU(), nn.Linear(24, 3))
            else:
                raise ValueError(kind)
            if kind != 'BiasPair':
                nn.init.zeros_(self.residual[-1].weight)
                nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def forward(self, batch, motion):
        with torch.no_grad():
            z = [self.base.encoders[m](batch[m]) for m in MODALITIES]
            context = torch.cat(z + [tokens(batch[m])[:, -1] for m in MODALITIES], -1)
            raw = self.base.head(context)
            available = torch.stack([batch[m]['valid'].flatten(1).any(1) for m in MODALITIES], 1).any(1)
            base_logits = torch.where(available[:, None], raw, self.base.prior_logits[None])
            coarse = torch.stack((base_logits[:, 0], base_logits[:, 1], torch.logsumexp(base_logits[:, 2:], -1)), -1)
        if self.kind == 'BiasPair':
            auxiliary = coarse + self.bias[None]
        elif self.kind == 'ContextPair':
            auxiliary = coarse + self.residual(context)
        else:
            extra = motion
            if self.kind == 'NoMotionPair':
                extra = motion.clone()
                extra[:, :5] = 0
            auxiliary = coarse + self.residual(torch.cat((context, extra), -1))
        # Internal coarse index 2 means Other, not original class 3.
        if not torch.isfinite(base_logits).all() or not torch.isfinite(auxiliary).all():
            raise RuntimeError('Nonfinite base or coarse scores before prediction')
        base_pred = base_logits.argmax(-1) + 1
        auxiliary_index = auxiliary.argmax(-1)
        gate = (base_pred <= 2) & (auxiliary_index < 2)
        prediction = torch.where(gate, auxiliary_index + 1, base_pred)
        return dict(base_logits=base_logits, auxiliary_logits=auxiliary, prediction=prediction)


def coarse_targets(original_labels_zero_based):
    return torch.where(original_labels_zero_based < 2, original_labels_zero_based,
                       torch.full_like(original_labels_zero_based, 2))
