"""Read-only damage pools and label-independent per-target augmentation."""
import json
from pathlib import Path

import numpy as np
import torch

from q3_sensor.protocol import MODALITIES
from q3_sensor.stress import stressed_modality

FIELDS = ('x', 'valid', 'fresh', 'age')


def train_damage(rows, frames, normalizer, modality, mode, config):
    if mode.startswith('burst') and mode != 'burst0':
        phase = int(mode[5:])
        modified = {cow: dict(values) for cow, values in frames.items()}
        for cow, values in frames.items():
            frame = values[modality]
            keep = np.ones(len(frame), bool)
            segments = {(r['start'], r['end']) for r in rows if r['cow_id'] == cow}
            for start, end in segments:
                inside = (frame.timestamp >= start) & (frame.timestamp < end)
                keep &= ~(inside & (((frame.timestamp - start + phase) % 300) < 30))
            modified[cow][modality] = frame.loc[keep]
        return stressed_modality(rows, modified, normalizer, modality, dict(kind='clean'))
    if mode == 'rate':
        condition = dict(kind='rate', factor=config['training_rate_factors'][modality])
    else:
        condition = {'clean': dict(kind='clean'), 'drop': dict(kind='drop'),
                     'delay15': dict(kind='delay', seconds=15), 'delay30': dict(kind='delay', seconds=30),
                     'burst0': dict(kind='burst')}[mode]
    return stressed_modality(rows, frames, normalizer, modality, condition)


class DamagePool:
    def __init__(self, directory, modes):
        directory = Path(directory)
        self.modes = modes if isinstance(modes, dict) else {m: modes for m in MODALITIES}
        self.arrays = {m: {mode: {key: np.load(directory / f'{m}_{mode}_{key}.npy', mmap_mode='r')
                                  for key in FIELDS} for mode in self.modes[m]} for m in MODALITIES}

    def batch(self, indices, choices=None, device='cuda'):
        if choices is None:
            choices = np.zeros((len(indices), 3), dtype=int)
        result = {}
        for at, modality in enumerate(MODALITIES):
            result[modality] = {}
            for key in FIELDS:
                example = self.arrays[modality][self.modes[modality][0]][key]
                value = np.empty((len(indices),) + example.shape[1:], example.dtype)
                for choice in np.unique(choices[:, at]):
                    selected = choices[:, at] == choice
                    value[selected] = self.arrays[modality][self.modes[modality][int(choice)]][key][indices[selected]]
                tensor = torch.from_numpy(value)
                result[modality][key] = (tensor.bool() if key == 'valid' else tensor).to(device)
        return result


def sample_damage(rng, count, config):
    modes = config['train_modes']
    regimes = config['augmentation_regimes']
    regime = rng.choice(3, size=count, p=[regimes['all_clean'], regimes['joint_delay30'], regimes['independent_per_modality']])
    choices = np.zeros((count, 3), dtype=np.int64)
    choices[regime == 1] = modes.index('delay30')
    independent = regime == 2
    choices[independent] = rng.choice(len(modes), size=(int(independent.sum()), 3),
                                     p=config['train_mode_probabilities'])
    return choices


def save_view(directory, modality, mode, data):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for key in FIELDS:
        np.save(directory / f'{modality}_{mode}_{key}.npy', data[key])


def load_conditions():
    return json.loads(Path('configs/sensor_stress_20261007.json').read_text())['conditions']


def evaluation_mode(condition, modality):
    if modality not in condition['modalities']:
        return 'clean'
    if condition['kind'] == 'rate':
        return f"rate{condition['factor']}"
    if condition['kind'] == 'delay':
        return f"delay{condition['seconds']}"
    return condition['kind']
