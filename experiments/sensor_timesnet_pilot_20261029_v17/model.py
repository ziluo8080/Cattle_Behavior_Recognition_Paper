"""Pinned official TimesNet backbone with the sealed matched-input fusion adapter.

FFT period selection intentionally depends on batch membership, as in upstream.
Record train/eval sample order, batch size and final-batch policy in run receipts.
"""
import ast
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from experiments.sensor_comparator_pilot_20261022_v2.models import (
    clean_inputs, fast_convolutions, pooled,
)


VENDOR = Path(__file__).parent / 'vendor' / 'Time-Series-Library'
DEFAULT_OPTIONS = dict(d_model=16, d_ff=32, e_layers=2, top_k=3,
                       num_kernels=4, dropout=.1)


def _definitions(path, namespace):
    """Execute definitions only; never install upstream's generic layers package."""
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    body = [node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)


@lru_cache(None)
def author_classes():
    namespace = dict(torch=torch, nn=nn, F=F, math=math, __name__=__name__)
    _definitions(VENDOR / 'layers' / 'Embed.py', namespace)
    _definitions(VENDOR / 'layers' / 'Conv_Blocks.py', namespace)
    _definitions(VENDOR / 'models' / 'TimesNet.py', namespace)
    return namespace


def author_config(channels, length, options):
    return SimpleNamespace(**options, task_name='classification', seq_len=length,
                           label_len=0, pred_len=0, enc_in=channels,
                           embed='timeF', freq='h', num_class=7)


class TimesNetEncoder(nn.Module):
    def __init__(self, channels, length, options):
        super().__init__()
        self.core = author_classes()['Model'](author_config(channels, length, options))
        # These task-specific parameters are unused by upstream classification's
        # x_mark=None path or superseded by our shared fusion readout.
        del self.core.projection
        del self.core.enc_embedding.temporal_embedding
        self.output = nn.Sequential(nn.Linear(options['d_model'], 64), nn.LayerNorm(64))
        self.register_buffer('positions', torch.arange(length))

    def forward_features(self, x, valid):
        z = self.core.enc_embedding(x, None)
        for block in self.core.model:
            z = self.core.layer_norm(block(z))
        # Upstream classification activation, dropout and padding mask.
        z = self.core.dropout(self.core.act(z))
        return z * valid.unsqueeze(-1)

    def forward(self, x, valid):
        return self.output(self.forward_features(x, valid)), valid, self.positions


class Fusion(nn.Module):
    def __init__(self, cfg, prior, fast=True):
        super().__init__()
        options = dict(DEFAULT_OPTIONS)
        options.update(cfg.get('timesnet', {}))
        self.options = options
        self.dynamic = TimesNetEncoder(8, 100, options)
        self.state = TimesNetEncoder(26, 60, options)
        self.context = nn.Sequential(nn.Linear(136, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(346, 128), nn.GELU(),
                                  nn.Dropout(.1), nn.Linear(128, 7))
        self.register_buffer('prior_logits',
                             torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8).log())
        if fast:
            # The shared helper assumes zero padding. Applying it to the author's
            # circular TokenEmbedding would silently change boundary behavior.
            for module in tuple(self.modules()):
                if type(module) is nn.Conv1d and module.padding_mode == 'zeros':
                    fast_convolutions(module)

    def forward(self, batch):
        dynamic, valid, state, state_valid = clean_inputs(batch)
        d, dm, dp = self.dynamic(dynamic, valid)
        s, sm, sp = self.state(state, state_valid)
        features = [pooled(d, dm, dp), pooled(d, dm, dp, 90),
                    pooled(s, sm, sp), pooled(s, sm, sp, 55),
                    self.context(batch['context']), batch['current']]
        logits = self.head(torch.cat(features, -1))
        observed = valid.any(1) | state_valid.any(1)
        return torch.where(observed[:, None], logits, self.prior_logits[None])


def make_model(cfg, prior):
    return Fusion(cfg, prior)
