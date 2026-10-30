"""Information controls without changing the frozen endpoint implementation."""
import numpy as np
import torch

from q3_sensor.endpoint_models import EndpointFusion, tokens
from q3_sensor.protocol import MODALITIES

FIELDS = ('x', 'valid', 'fresh', 'age')


def repeat_current(batch):
    return {m: {k: batch[m][k][:, -1:, :].expand(-1, 60, -1)
                for k in FIELDS} for m in MODALITIES}


class CurrentRepeatGRU(EndpointFusion):
    def __init__(self, prior):
        super().__init__('gru', prior)

    def forward(self, batch):
        return super().forward(repeat_current(batch))


def window_rf_features(batch):
    parts = []
    for m in MODALITIES:
        data = batch[m]
        dims = data['x'].shape[-1]
        encoded = tokens(data)
        parts.extend(encoded[..., start:end].reshape(len(encoded), -1).numpy()
                     for start, end in ((0, dims), (dims, 2 * dims),
                                        (2 * dims, 2 * dims + 1),
                                        (2 * dims + 1, 2 * dims + 2)))
    result = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
    if result.shape[1] != 1560 or not np.isfinite(result).all():
        raise AssertionError('Invalid full-window RF features')
    return result


def available_rows(batch, endpoint=False):
    values = [data['valid'][:, -1].any(-1) if endpoint else data['valid'].flatten(1).any(-1)
              for data in batch.values()]
    return torch.stack(values, 1).any(1).numpy()
