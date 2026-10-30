"""Matched-input comparators with pinned author backbones and declared adapters."""
import ast
import copy
import math
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

VENDOR = Path(__file__).parent / 'vendor'


class FastConv1d(nn.Conv1d):
    """Equivalent height-one channels-last Conv2d; parameters keep Conv1d shapes."""
    def forward(self, x):
        return F.conv2d(x.unsqueeze(2).contiguous(memory_format=torch.channels_last),
                        self.weight.unsqueeze(2), self.bias, (1, self.stride[0]),
                        (0, self.padding[0]), (1, self.dilation[0]), self.groups).squeeze(2)


def fast_convolutions(model):
    for module in model.modules():
        if type(module) is nn.Conv1d:
            module.__class__ = FastConv1d


class DropPath(nn.Module):
    """Per-example stochastic depth with the same keep-probability scaling."""
    def __init__(self, probability=0.):
        super().__init__(); self.probability = probability

    def forward(self, x):
        if not self.training or self.probability == 0:
            return x
        keep = 1 - self.probability
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


def definitions(path, namespace, names=None):
    """Load model definitions only, never author training loops or import side effects."""
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and (names is None or n.name in names)]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), 'exec'), namespace)


def deterministic_spectral_mask(self, x_fft):
    """Author's lower-median value with deterministic CUDA ordering and tie handling."""
    energy = torch.abs(x_fft).pow(2).sum(dim=-1)
    ordered = energy.sort(dim=1, stable=True).values
    median = ordered[:, (ordered.shape[1]-1)//2, None]
    normalized = energy / (median + 1e-6)
    mask = ((normalized > self.threshold_param).float()-self.threshold_param).detach()+self.threshold_param
    return mask.unsqueeze(-1)


@lru_cache(None)
def author_classes():
    base = dict(torch=torch, nn=nn, F=F, math=math, __name__=__name__)
    modern = base.copy()
    prefix = VENDOR / 'moderntcn/ModernTCN-classification/models'
    definitions(prefix / 'ModernTCN_Layer.py', modern)
    definitions(prefix / 'ModernTCN.py', modern)
    tsla = base.copy()
    tsla.update(L=SimpleNamespace(LightningModule=nn.Module), DropPath=DropPath,
                trunc_normal_=nn.init.trunc_normal_,
                args=SimpleNamespace(ICB=True, ASB=True, adaptive_filter=True))
    definitions(VENDOR / 'tslanet/Classification/TSLANet_classification.py', tsla,
                {'ICB', 'PatchEmbed', 'Adaptive_Spectral_Block', 'TSLANet_layer'})
    tsla['Adaptive_Spectral_Block'].create_adaptive_high_freq_mask = deterministic_spectral_mask
    return modern, tsla


def pooled(sequence, valid, positions, start=0):
    usable = valid & (positions[None] >= start)
    values = torch.where(usable[:, :, None], sequence, 0.)
    return values.sum(1) / usable.sum(1, keepdim=True).clamp_min(1)


class RecurrentEncoder(nn.Module):
    def __init__(self, channels, length, cfg):
        super().__init__()
        self.project = nn.Linear(channels, cfg['projection'])
        self.gru = nn.GRU(cfg['projection'], cfg['hidden'], cfg['layers'], batch_first=True)
        self.output = nn.Sequential(nn.Linear(cfg['hidden'], 64), nn.LayerNorm(64))
        self.register_buffer('positions', torch.arange(length))

    def forward(self, x, valid):
        self.gru.flatten_parameters()
        z, _ = self.gru(F.gelu(self.project(x)))
        return self.output(z), valid, self.positions


class AttentionEncoder(nn.Module):
    def __init__(self, channels, length, cfg):
        super().__init__(); width = cfg['width']
        self.project = nn.Linear(channels, width)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(width, cfg['heads'], cfg['ff'],
            cfg['dropout'], activation='gelu', batch_first=True, norm_first=True)
            for _ in range(cfg['layers'])])
        position = torch.arange(length)[:, None]
        angles = position * torch.exp(torch.arange(0, width, 2) * (-math.log(10000) / width))
        encoding = torch.zeros(length, width)
        encoding[:, 0::2], encoding[:, 1::2] = angles.sin(), angles.cos()
        self.register_buffer('encoding', encoding)
        self.register_buffer('positions', torch.arange(length))
        self.output = nn.LayerNorm(64)

    def forward(self, x, valid):
        z = self.project(x) + self.encoding
        usable = valid.clone(); usable[:, 0] |= ~usable.any(1)
        for layer in self.layers:
            z = layer(z, src_key_padding_mask=~usable)
        return self.output(z), valid, self.positions


class ModernEncoder(nn.Module):
    def __init__(self, channels, length, cfg):
        super().__init__(); modern, _ = author_classes()
        self.core = modern['ModernTCN'](task_name='classification', patch_size=cfg['patch_size'],
            patch_stride=cfg['patch_stride'], stem_ratio=2, downsample_ratio=2,
            ffn_ratio=cfg['ffn_ratio'], num_blocks=cfg['blocks'], large_size=cfg['large_kernels'],
            small_size=cfg['small_kernels'], dims=cfg['dims'], dw_dims=cfg['dims'], nvars=channels,
            c_in=[channels], seq_len=length, use_multi_scale=False, revin=False,
            backbone_dropout=cfg['dropout'], target_window=1, class_num=7)
        # Author's task-specific heads are replaced by the common task adapter.
        del self.core.head; del self.core.head_class
        self.output = nn.Sequential(nn.Linear(channels * cfg['dims'][-1], 64), nn.LayerNorm(64))
        self.length = length

    def forward(self, x, valid):
        z = self.core.forward_feature(x.transpose(1, 2))
        z = z.permute(0, 3, 1, 2).flatten(2)
        mask = F.max_pool1d(F.pad(valid[:, None].float(), (0, 4), mode='replicate'), 8, 4)
        if mask.shape[-1] % 2:
            mask = F.pad(mask, (0, 1), mode='replicate')
        mask = F.max_pool1d(mask, 2, 2)[:, 0].bool()
        positions = (torch.arange(z.shape[1], device=x.device) * 8 + 11).clamp_max(self.length-1)
        assert mask.shape[1] == z.shape[1]
        return self.output(z), mask, positions


class SpectralEncoder(nn.Module):
    def __init__(self, channels, length, cfg):
        super().__init__(); _, tsla = author_classes(); width = cfg['width']
        self.patch = tsla['PatchEmbed'](length, cfg['patch_size'], channels, width)
        self.position = nn.Parameter(torch.zeros(1, self.patch.num_patches, width))
        self.dropout = nn.Dropout(cfg['dropout'])
        self.blocks = nn.ModuleList([tsla['TSLANet_layer'](width, drop=cfg['dropout'],
            drop_path=rate) for rate in torch.linspace(0, cfg['dropout'], cfg['depth']).tolist()])
        self.output = nn.Sequential(nn.Linear(width, 64), nn.LayerNorm(64))
        self.patch_size = cfg['patch_size']; self.stride = self.patch_size // 2
        self.register_buffer('positions', torch.arange(self.patch.num_patches) * self.stride + self.patch_size-1)
        nn.init.trunc_normal_(self.position, std=.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=.02)
                if module.bias is not None: nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.zeros_(module.bias); nn.init.ones_(module.weight)

    def forward(self, x, valid):
        z = self.dropout(self.patch(x.transpose(1, 2)) + self.position)
        for block in self.blocks: z = block(z)
        mask = valid.unfold(1, self.patch_size, self.stride).any(-1)
        return self.output(z), mask, self.positions


def clean_inputs(batch):
    valid = batch['dynamic'][:, :, 6] > .5
    physical = torch.where(valid[:, :, None], batch['dynamic'][:, :, :6], 0.)
    age = torch.where(valid[:, :, None], batch['dynamic'][:, :, 7:].clamp(0, .2), .2)
    dynamic = torch.cat((physical, valid[:, :, None].float(), age), -1)
    state_valid = batch['state_valid'].bool()
    state = torch.where(state_valid[:, :, None], batch['state'], 0.)
    return dynamic, valid, state, state_valid


class Fusion(nn.Module):
    def __init__(self, kind, cfg, prior, fast=True):
        super().__init__()
        encoder, options = {
            'GRU_Fusion': (RecurrentEncoder, cfg['gru']),
            'Transformer_Fusion': (AttentionEncoder, cfg['transformer']),
            'ModernTCN_Fusion': (ModernEncoder, cfg['moderntcn']),
            'TSLANet_Sup_Fusion': (SpectralEncoder, cfg['tslanet']),
        }[kind]
        self.dynamic, self.state = encoder(8, 100, options), encoder(26, 60, options)
        self.context = nn.Sequential(nn.Linear(136, 64), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(346, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 7))
        self.register_buffer('prior_logits', torch.as_tensor(prior, dtype=torch.float32).clamp_min(1e-8).log())
        if fast: fast_convolutions(self)

    def forward(self, batch):
        dynamic, valid, state, state_valid = clean_inputs(batch)
        d, dm, dp = self.dynamic(dynamic, valid)
        s, sm, sp = self.state(state, state_valid)
        features = [pooled(d, dm, dp), pooled(d, dm, dp, 90),
                    pooled(s, sm, sp), pooled(s, sm, sp, 55),
                    self.context(batch['context']), batch['current']]
        logits = self.head(torch.cat(features, -1))
        return torch.where((valid.any(1) | state_valid.any(1))[:, None], logits, self.prior_logits[None])


def make_model(arm, cfg, prior):
    return Fusion(arm, cfg, prior)


def synthetic_batch(device='cpu', count=4):
    batch = dict(dynamic=torch.randn(count, 100, 8, device=device),
                 state=torch.randn(count, 60, 26, device=device),
                 state_valid=torch.ones(count, 60, dtype=torch.bool, device=device),
                 context=torch.randn(count, 136, device=device), current=torch.randn(count, 26, device=device))
    batch['dynamic'][:, :, 6] = 1.; batch['dynamic'][:, :, 7] = .05
    batch['current'] = batch['state'][:, -1].clone()
    return batch


def synthetic_check(cfg, device='cpu'):
    torch.set_num_threads(4); torch.manual_seed(17)
    torch.backends.mha.set_fastpath_enabled(False)
    # Verify author's median value (including even lengths/ties) without executing its unsupported CUDA op.
    for length in (8, 13):
        values = torch.randn(4, length, 3, device=device, dtype=torch.complex64)
        values[0] = 0.
        layer = SimpleNamespace(threshold_param=torch.tensor([.4], device=device))
        energy = values.abs().pow(2).sum(-1)
        median_cpu = energy.cpu().median(dim=1, keepdim=True).values.to(device)
        normalized = energy / (median_cpu+1e-6)
        expected = (((normalized > layer.threshold_param).float()-layer.threshold_param).detach()+layer.threshold_param).unsqueeze(-1)
        torch.testing.assert_close(deterministic_spectral_mask(layer, values), expected, rtol=0, atol=0)
    records = []
    for arm in cfg['arms']:
        model = Fusion(arm, cfg, [1/7]*7, fast=False).to(device)
        fast = copy.deepcopy(model); fast_convolutions(fast)
        batch = synthetic_batch(device)
        model.eval(); fast.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(batch), fast(batch), rtol=2e-4, atol=2e-5)
        model.train(); fast.train()
        # Nontrivial CE witnesses all active feature paths, not only output shapes.
        torch.manual_seed(23); loss = F.cross_entropy(fast(batch), torch.tensor([0, 1, 5, 6], device=device))
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.parameters())
        missing = copy.deepcopy(batch); missing['dynamic'][:, :, 6] = 0.; missing['state_valid'][:] = False
        fast.eval()
        with torch.no_grad():
            torch.testing.assert_close(fast(missing), fast.prior_logits[None].expand(4, -1), rtol=0, atol=0)
            # Unavailable physical payload must not alter predictions.
            other = copy.deepcopy(missing); other['dynamic'][:, :, :6] = 1e9; other['state'][:] = -1e9
            torch.testing.assert_close(fast(missing), fast(other), rtol=0, atol=0)
        records.append(dict(arm=arm, parameters=sum(p.numel() for p in fast.parameters()),
            finite_gradients=True, fast_conv_equivalence=True, missing_fallback=True))
    return dict(status='PASS', scope='synthetic only, no scientific data or scores',
                deterministic_spectral_lower_median_exact=True, records=records)
