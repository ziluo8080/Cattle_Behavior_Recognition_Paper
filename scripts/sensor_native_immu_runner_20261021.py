"""Synchronized selection-group state and identity-safe resume primitives."""
from dataclasses import asdict, dataclass, field
from decimal import Decimal
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

from scripts.sensor_native_immu_selection_policy_20261021 import SEEDS


@dataclass
class GroupState:
    common_epoch: int = 0
    best_monitor: str = '-1'
    stale_epochs: int = 0
    raw_means: list = field(default_factory=list)
    ema_means: list = field(default_factory=list)
    seed_records: dict = field(default_factory=lambda: {seed: [] for seed in SEEDS})

    def to_dict(self):
        return asdict(self)


def checkpoint_identity(protocol, fold, arm, lr_index, config_sha256):
    if protocol not in ('TS', 'OS') or fold not in {f'fold_{i}' for i in range(1, 6)}:
        raise ValueError('Invalid selection group')
    if lr_index not in (0, 1) or not arm or not config_sha256:
        raise ValueError('Incomplete immutable group identity')
    return dict(protocol=protocol, fold=fold, arm=arm, lr_index=int(lr_index),
                config_sha256=config_sha256, seeds=list(SEEDS))


def mark_epoch(state, epoch, seed_records, raw_mean, ema_mean):
    if epoch != state.common_epoch + 1:
        raise ValueError('Only the next common epoch can be committed')
    if set(seed_records) != set(SEEDS):
        raise ValueError('All three synchronized seed records are required')
    raw, ema = Decimal(str(raw_mean)), Decimal(str(ema_mean))
    if any(not x.is_finite() or not Decimal(0) <= x <= Decimal(1) for x in (raw, ema)):
        raise ValueError('Macro-F1 means must be finite values in [0,1]')
    for seed in SEEDS:
        record = seed_records[seed]
        if not isinstance(record, dict) or record.get('epoch') != epoch or record.get('seed') != seed:
            raise ValueError('Seed record identity or epoch mismatch')
    monitor = max(raw, ema)
    previous = Decimal(state.best_monitor)
    state.stale_epochs = 0 if monitor > previous else state.stale_epochs + 1
    state.best_monitor = str(max(previous, monitor))
    state.common_epoch = epoch
    state.raw_means.append(str(raw))
    state.ema_means.append(str(ema))
    for seed in SEEDS:
        state.seed_records[seed].append(seed_records[seed])
    return state


def restore_checkpoint(payload, expected_identity):
    if payload.get('identity') != expected_identity:
        raise ValueError('Checkpoint identity differs from the sealed selection group')
    raw = payload.get('state')
    if not isinstance(raw, dict):
        raise ValueError('Checkpoint lacks group state')
    state = GroupState(**raw)
    if set(state.seed_records) != set(SEEDS):
        raise ValueError('Checkpoint seed roster changed')
    if state.common_epoch != len(state.raw_means) or state.common_epoch != len(state.ema_means):
        raise ValueError('Checkpoint common history is incomplete')
    if any(len(state.seed_records[s]) != state.common_epoch for s in SEEDS):
        raise ValueError('Checkpoint does not contain synchronized seed histories')
    for seed in SEEDS:
        for epoch, record in enumerate(state.seed_records[seed], 1):
            if record.get('seed') != seed or record.get('epoch') != epoch:
                raise ValueError('Checkpoint seed history order is corrupt')
    return state


def save_checkpoint(directory, payload):
    """Immutable generations; atomically point LATEST at a fully durable state."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = f'state_{time.time_ns()}.pkl'
    data = pickle.dumps(payload, protocol=5)
    with (directory / name).open('xb') as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    receipt = dict(file=name, sha256=hashlib.sha256(data).hexdigest())
    temporary = directory / 'LATEST.tmp'
    temporary.write_text(json.dumps(receipt), encoding='utf-8')
    temporary.replace(directory / 'LATEST.json')
    return receipt


def load_checkpoint(directory, identity):
    """Only local trusted checkpoints; check payload hash before deserializing."""
    directory = Path(directory)
    receipt = json.loads((directory / 'LATEST.json').read_text())
    name = receipt['file']
    if Path(name).name != name or not name.startswith('state_') or not name.endswith('.pkl'):
        raise ValueError('Checkpoint generation path is invalid')
    data = (directory / name).read_bytes()
    if hashlib.sha256(data).hexdigest() != receipt['sha256']:
        raise ValueError('Checkpoint generation hash differs')
    payload = pickle.loads(data)
    restore_checkpoint(payload, identity)
    return payload


def scores_from_predictions(labels, predictions):
    """Count all seven classes without sklearn or model-side metric state."""
    labels, predictions = list(labels), list(predictions)
    if not labels or len(labels) != len(predictions):
        raise ValueError('Nonempty matching predictions required')
    cm = [[0] * 7 for _ in range(7)]
    for actual, predicted in zip(labels, predictions):
        if actual not in range(1, 8) or predicted not in range(1, 8):
            raise ValueError('Labels/predictions must be in 1..7')
        cm[int(actual) - 1][int(predicted) - 1] += 1
    support = [sum(row) for row in cm]
    predicted_count = [sum(row[i] for row in cm) for i in range(7)]
    f1 = [2 * cm[i][i] / (support[i] + predicted_count[i]) if support[i] + predicted_count[i] else 0. for i in range(7)]
    return dict(n=len(labels), macro_f1=sum(f1) / 7, f1=f1, support=support, confusion_matrix=cm)
