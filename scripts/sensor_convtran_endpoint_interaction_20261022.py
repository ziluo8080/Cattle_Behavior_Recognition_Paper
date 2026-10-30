"""Experimental endpoint-query residual fusion; no experiment launcher.

Reuses the sealed two-rate encoders and head without editing the old model.
Zero output projections give the concat function at initialization in eval mode.
This is not a guarantee of preserving a trained classifier during optimization.
"""
import torch
from torch import nn

from scripts.sensor_convtran_context_model_20261022 import (
    TwoRateConvTran, masked_mean, safe_attention,
)


class EndpointResidual(nn.Module):
    """Endpoint or token queries read the other branch with the same parameters."""

    def __init__(self, width=32, heads=4, dropout=.1):
        super().__init__()
        if width % heads:
            raise ValueError('Width must be divisible by heads')
        self.heads, self.scale = heads, (width // heads) ** -.5
        self.query_norm = nn.LayerNorm(width)
        self.memory_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width, bias=False)
        nn.init.zeros_(self.output.weight)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, query, memory, query_valid, memory_valid):
        batch, length, width = memory.shape
        single = query.ndim == 2
        if single:
            query, query_valid = query[:, None], query_valid[:, None]
        qlength = query.shape[1]
        # Exclude invalid values before normalization as well as in attention.
        query = torch.where(query_valid[:, :, None], query, 0.)
        memory = torch.where(memory_valid[:, :, None], memory, 0.)
        q = self.query(self.query_norm(query)).view(batch, qlength, self.heads, -1).transpose(1, 2)
        normalized = self.memory_norm(memory)
        k = self.key(normalized).view(batch, length, self.heads, -1).transpose(1, 2)
        v = self.value(normalized).view(batch, length, self.heads, -1).transpose(1, 2)
        attention = safe_attention((q @ k.transpose(-1, -2)) * self.scale, memory_valid)
        mixed = (self.attention_dropout(attention) @ v).transpose(1, 2).reshape(batch, qlength, width)
        delta = self.output_dropout(self.output(mixed))
        supported = query_valid & memory_valid.any(1, keepdim=True)
        delta = torch.where(supported[:, :, None], delta, 0.)
        return delta[:, 0] if single else delta


class EndpointInteractionConvTran(TwoRateConvTran):
    """Keep pre-interaction global readouts; add residuals to endpoint readouts."""

    def __init__(self, cfg, prior):
        # Instantiate all common modules first, in exactly the concat order.
        super().__init__(cfg, prior, interaction=False)
        self.endpoint_queries = True
        self.dynamic_endpoint_from_state = EndpointResidual(cfg['width'], cfg['heads'], cfg['dropout'])
        self.state_endpoint_from_dynamic = EndpointResidual(cfg['width'], cfg['heads'], cfg['dropout'])

    def forward(self, batch, diagnostics=False):
        dynamic_valid = batch['dynamic'][:, :, 6] > .5
        physical = torch.where(dynamic_valid[:, :, None], batch['dynamic'][:, :, :6], 0.)
        age = torch.where(dynamic_valid[:, :, None], batch['dynamic'][:, :, 7:].clamp(0, .2), .2)
        cleaned = torch.cat((physical, dynamic_valid[:, :, None].float(), age), -1)
        dynamic = self.dynamic(cleaned, dynamic_valid)
        state_valid = batch['state_valid'].bool()
        state = self.state(batch['state'], state_valid)

        dg, de = masked_mean(dynamic, dynamic_valid), masked_mean(dynamic, dynamic_valid, 90)
        sg, se = masked_mean(state, state_valid), masked_mean(state, state_valid, 55)
        dynamic_query_valid = dynamic_valid[:, 90:].any(1)
        state_query_valid = state_valid[:, 55:].any(1)
        # Both directions use the original encoded sequences and readouts.
        if self.endpoint_queries:
            rd = self.dynamic_endpoint_from_state(de, state, dynamic_query_valid, state_valid)
            rs = self.state_endpoint_from_dynamic(se, dynamic, state_query_valid, dynamic_valid)
            readouts = (dg, de + rd, sg, se + rs)
        else:
            rd = self.dynamic_endpoint_from_state(dynamic, state, dynamic_valid, state_valid)
            rs = self.state_endpoint_from_dynamic(state, dynamic, state_valid, dynamic_valid)
            revised_d, revised_s = dynamic + rd, state + rs
            readouts = (masked_mean(revised_d, dynamic_valid), masked_mean(revised_d, dynamic_valid, 90),
                        masked_mean(revised_s, state_valid), masked_mean(revised_s, state_valid, 55))
        features = torch.cat((*readouts,
                              self.context(batch['context']), batch['current']), -1)
        logits = self.head(features)
        available = dynamic_valid.any(1) | state_valid.any(1)
        logits = torch.where(available[:, None], logits, self.prior_logits[None])
        if diagnostics:
            return logits, dict(dynamic_delta=rd, state_delta=rs,
                                dynamic_endpoint=de, state_endpoint=se,
                                dynamic_query_valid=dynamic_query_valid,
                                state_query_valid=state_query_valid)
        return logits


class FullSequenceResidualInteractionConvTran(EndpointInteractionConvTran):
    """Control A: same residual module, all-token queries and updated global reads."""

    def __init__(self, cfg, prior):
        super().__init__(cfg, prior)
        self.endpoint_queries = False
