"""Small controlled residual adapters; original sensor model files stay frozen."""
import torch
from torch import nn
from torch.nn import functional as F
from q3_sensor.endpoint_models import EndpointFusion, tokens
from q3_sensor.model import CausalConv
from q3_sensor.protocol import MODALITIES


class ResidualSensor(nn.Module):
    def __init__(self, kind, prior, base_initial, branch_seed):
        super().__init__()
        self.kind = kind
        self.base = EndpointFusion('gru', prior)
        self.base.load_state_dict(base_initial, strict=True)
        # Auxiliary initialization does not consume the main training random stream.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(branch_seed)
            if kind == 'DenseResidual':
                self.residual = nn.Sequential(nn.Linear(218, 24), nn.GELU(), nn.Linear(24, 7))
            elif kind in ['StaticLocal', 'LocalTCN']:
                self.local = nn.ModuleDict({m: nn.Sequential(CausalConv(32, 16, 1), CausalConv(16, 16, 2))
                                            for m in ('uwb', 'hd')})
                self.residual = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 7))
            elif kind != 'Base':
                raise ValueError(kind)
            if kind != 'Base':
                nn.init.zeros_(self.residual[-1].weight)
                nn.init.zeros_(self.residual[-1].bias)

    def forward(self, batch):
        if self.kind == 'Base':
            return self.base(batch)
        representations, local = [], []
        for modality in MODALITIES:
            data, encoder = batch[modality], self.base.encoders[modality]
            endpoint_valid = data['valid'][:, -1].any(-1, keepdim=True)
            projected = F.gelu(encoder.project(tokens(data)))
            sequence, _ = encoder.gru(projected)
            representations.append(encoder.norm(sequence[:, -1]) * endpoint_valid)
            if self.kind in ['StaticLocal', 'LocalTCN'] and modality in self.local:
                branch = projected
                if self.kind == 'StaticLocal':
                    branch = projected[:, -1:, :].expand(-1, projected.shape[1], -1)
                local.append(self.local[modality](branch)[:, -1] * endpoint_valid)
        context = torch.cat(representations + [tokens(batch[m])[:, -1] for m in MODALITIES], -1)
        addition = self.residual(context if self.kind == 'DenseResidual' else torch.cat(local, -1))
        logits = self.base.head(context) + addition
        available = torch.stack([batch[m]['valid'].flatten(1).any(1) for m in MODALITIES], 1).any(1)
        return torch.where(available[:, None], logits, self.base.prior_logits[None])
