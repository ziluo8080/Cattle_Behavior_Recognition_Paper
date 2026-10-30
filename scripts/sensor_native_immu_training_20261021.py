"""One-framework-at-a-time TRAIN/VAL engine with complete epoch checkpoint state."""
import copy
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np

from scripts.sensor_native_immu_runner_20261021 import (
    GroupState, load_checkpoint, mark_epoch, save_checkpoint, scores_from_predictions,
)
from scripts.sensor_native_immu_selection_policy_20261021 import SEEDS, synchronized_epoch_rows, should_stop_group

KINDS = {
    'HInception_Native_FeatureFusion': ('fusion', 'native', .2),
    'HInception_LowFreq_FeatureFusion': ('fusion', 'low_frequency', 1.),
    'HInception_DynamicOnly': ('dynamic_only', 'native', .2),
    'ContextOnly': ('context_only', 'native', .2),
    'HInception_LateLogits': ('late_logits', 'native', .2),
    'ConvTran_Native_FeatureFusion': ('convtran', 'native', .2),
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        while chunk := file.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_selection(root, protocol, fold):
    """Load only hash-bound selection targets, with no formal/Test loader path."""
    if protocol not in ('TS', 'OS') or fold not in {f'fold_{i}' for i in range(1, 6)}:
        raise ValueError('Invalid protocol/fold')
    root = Path(root).resolve()
    directory = root / 'reports/sensor_native_immu_preflight_20261021_v2/data' / protocol / fold
    receipt = json.loads((directory / 'DATA_RECEIPT.json').read_text())
    if receipt['phase'] != 'selection':
        raise ValueError('TRAIN/VAL selection inputs required')
    for name, digest in receipt['outputs'].items():
        if sha(directory / name) != digest:
            raise ValueError(f'Selection cache hash changed: {name}')
    path = (root / receipt['native_inputs_parent']).resolve()
    allowed = root / 'reports/sensor_native_immu_preflight_20261021_v1/data' / protocol / fold / 'inputs.npz'
    if path != allowed or sha(path) != receipt['native_inputs_parent_sha256']:
        raise ValueError('Native input cache identity differs')
    rows = [json.loads(line) for line in (directory / 'targets.jsonl').read_text().splitlines()]
    if {r['split'] for r in rows} != {'train', 'val'}:
        raise ValueError('Only TRAIN and VAL roles allowed')
    y = np.asarray([r['label'] - 1 for r in rows], np.int64)
    if not np.isin(y, np.arange(7)).all():
        raise ValueError('Seven behavior labels required')
    train = np.asarray([r['split'] == 'train' for r in rows])
    with np.load(path, allow_pickle=False) as arrays:
        if not np.array_equal(train, arrays['train_mask']):
            raise ValueError('Target role mask changed')
        data = dict(native=arrays['native'], low_frequency=arrays['low_frequency'])
    data.update(context=np.load(directory / 'context.npy', allow_pickle=False), labels=y,
                train=np.flatnonzero(train), evaluation=np.flatnonzero(~train),
                receipt_sha256=sha(directory / 'DATA_RECEIPT.json'), protocol=protocol, fold=fold)
    return data


class SeedTrack:
    """Own model, optimizer, EMA and RNG state for exactly one seed."""
    def __init__(self, arm, seed, data, learning_rate=.001):
        if arm not in KINDS or seed not in SEEDS:
            raise ValueError('Unregistered arm/seed')
        self.arm, self.seed = arm, seed
        self.kind, self.dynamic_name, max_age = KINDS[arm]
        self.steps = 0
        counts = np.bincount(data['labels'][data['train']], minlength=7)
        if (counts == 0).any():
            raise ValueError('All seven TRAIN classes required')
        self.weights = np.sqrt(counts.sum() / (7 * counts)).astype(np.float32)
        self.torch = None
        if self.kind == 'convtran':
            import torch
            from scripts.sensor_native_convtran_20261021 import NativeConvTran
            from scripts.sensor_uwb_event_motion_20261020 import ParameterEMA
            self.torch = torch
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            self.model = NativeConvTran().cuda()
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate,
                                               weight_decay=.01, eps=1e-8)
            self.ema = ParameterEMA(self.model)
            self.rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
        else:
            import tensorflow as tf
            from scripts.sensor_native_immu_model_20261021_v2 import make_model, ParameterEMA
            self.tf = tf
            tf.keras.utils.set_random_seed(seed)
            self.model = make_model(self.kind, max_age=max_age, seed=seed)
            self.optimizer = tf.keras.optimizers.AdamW(learning_rate, weight_decay=.01,
                                                       epsilon=1e-8, jit_compile=False)
            self.optimizer.build(self.model.trainable_variables)
            self.ema = ParameterEMA(self.model)
            self.tf_step = tf.function(self._tf_step, jit_compile=False, reduce_retracing=True)

    def _tf_step(self, dynamic, context, labels):
        tf = self.tf
        from scripts.sensor_native_immu_model_20261021 import weighted_cross_entropy
        with tf.GradientTape() as tape:
            logits = self.model(dict(dynamic=dynamic, context=context), training=True)
            loss = weighted_cross_entropy(labels, logits, self.weights)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        for gradient in gradients:
            if gradient is None:
                raise ValueError('Disconnected trainable variable')
            tf.debugging.assert_all_finite(gradient, 'Nonfinite training gradient')
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss

    def batch(self, data, ids):
        return dict(dynamic=data[self.dynamic_name][ids], context=data['context'][ids])

    def step(self, data, ids):
        batch = self.batch(data, ids)
        if self.torch is None:
            loss = float(self.tf_step(batch['dynamic'], batch['context'], data['labels'][ids]).numpy())
        else:
            torch = self.torch
            torch.set_rng_state(self.rng[0]); torch.cuda.set_rng_state_all(self.rng[1])
            self.model.train(); self.optimizer.zero_grad(set_to_none=True)
            batch = {k: torch.as_tensor(v, dtype=torch.float32, device='cuda') for k, v in batch.items()}
            logits = self.model(batch)
            target = torch.as_tensor(data['labels'][ids], dtype=torch.int64, device='cuda')
            loss_tensor = torch.nn.functional.cross_entropy(logits, target,
                               weight=torch.as_tensor(self.weights, device='cuda'))
            loss_tensor.backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in self.model.parameters()):
                raise ValueError('Nonfinite/disconnected Torch gradient')
            self.optimizer.step()
            self.rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all())
            loss = float(loss_tensor.detach().cpu())
        if not np.isfinite(loss):
            raise ValueError('Nonfinite loss')
        self.ema.update(self.model) if self.torch is not None else self.ema.update()
        self.steps += 1
        return loss

    def predict(self, data, ids, view, batch_size=256):
        if view not in ('raw', 'ema'):
            raise ValueError('Unknown model view')
        pieces = []
        if self.torch is None:
            from contextlib import nullcontext
            scope = self.ema.averaged_weights() if view == 'ema' else nullcontext(self.model)
            with scope:
                for start in range(0, len(ids), batch_size):
                    logits = self.model(self.batch(data, ids[start:start+batch_size]), training=False).numpy()
                    if not np.isfinite(logits).all():
                        raise ValueError('Nonfinite prediction')
                    pieces.append(logits.argmax(1) + 1)
        else:
            torch = self.torch
            model = self.ema.model if view == 'ema' else self.model
            model.eval()
            with torch.no_grad():
                for start in range(0, len(ids), batch_size):
                    batch = {k: torch.as_tensor(v, dtype=torch.float32, device='cuda')
                             for k, v in self.batch(data, ids[start:start+batch_size]).items()}
                    logits = model(batch)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite prediction')
                    pieces.append(logits.argmax(1).cpu().numpy() + 1)
        return np.concatenate(pieces)

    def epoch(self, data, epoch, batch_size=256):
        order = np.random.default_rng(np.random.SeedSequence([self.seed, epoch])).permutation(data['train'])
        numerator = denominator = 0.
        for start in range(0, len(order), batch_size):
            ids = order[start:start+batch_size]
            loss = self.step(data, ids)
            weight = float(self.weights[data['labels'][ids]].sum())
            numerator += loss * weight; denominator += weight
        evaluation = data['evaluation']
        scores = {view: scores_from_predictions((data['labels'][evaluation]+1).tolist(),
                    self.predict(data, evaluation, view).tolist()) for view in ('raw', 'ema')}
        return dict(seed=self.seed, epoch=epoch, train_loss=numerator / denominator,
                    steps=self.steps, validation=scores)

    def export(self):
        if self.torch is None:
            payload = dict(model=[w.numpy() for w in self.model.weights],
                           optimizer=[w.numpy() for w in self.optimizer.variables()],
                           EMA=[w.numpy() for w in self.ema.shadow])
        else:
            torch = self.torch
            def cpu(value):
                if torch.is_tensor(value):
                    return value.detach().cpu().clone()
                if isinstance(value, dict):
                    return {k: cpu(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [cpu(v) for v in value]
                return copy.deepcopy(value)
            payload = dict(model=cpu(self.model.state_dict()), optimizer=cpu(self.optimizer.state_dict()),
                           EMA=cpu(self.ema.model.state_dict()), rng=cpu(self.rng))
        return dict(payload, steps=self.steps, EMA_updates=self.ema.updates,
                    arm=self.arm, seed=self.seed, class_weights=self.weights.copy())

    def restore(self, payload):
        if payload['arm'] != self.arm or payload['seed'] != self.seed or not np.array_equal(payload['class_weights'], self.weights):
            raise ValueError('Seed checkpoint identity differs')
        if self.torch is None:
            for variables, arrays in ((self.model.weights, payload['model']),
                                       (self.optimizer.variables(), payload['optimizer']),
                                       (self.ema.shadow, payload['EMA'])):
                if len(variables) != len(arrays):
                    raise ValueError('Checkpoint variable count differs')
                for variable, array in zip(variables, arrays):
                    if tuple(variable.shape) != array.shape:
                        raise ValueError('Checkpoint variable shape differs')
                    variable.assign(array)
        else:
            self.model.load_state_dict(payload['model']); self.optimizer.load_state_dict(payload['optimizer'])
            self.ema.model.load_state_dict(payload['EMA']); self.rng = payload['rng']
        self.steps, self.ema.updates = payload['steps'], payload['EMA_updates']


def run_group_epochs(data, identity, directory, until_epoch, arm, learning_rate=.001, batch_size=256,
                     minimum_epochs=20, patience=10, maximum_epochs=100):
    """Persist only complete common epochs; interrupted partial epochs are replayed."""
    if identity['arm'] != arm or identity['protocol'] != data['protocol'] or identity['fold'] != data['fold']:
        raise ValueError('Data/group identity mismatch')
    if not 1 <= until_epoch <= maximum_epochs or batch_size < 1:
        raise ValueError('Invalid group budget')
    state = GroupState()
    tracks = {seed: SeedTrack(arm, seed, data, learning_rate) for seed in SEEDS}
    if (Path(directory) / 'LATEST.json').exists():
        payload = load_checkpoint(directory, identity)
        from scripts.sensor_native_immu_runner_20261021 import restore_checkpoint
        state = restore_checkpoint(payload, identity)
        for seed, track in tracks.items():
            track.restore(payload['tracks'][seed])
    for epoch in range(state.common_epoch + 1, until_epoch + 1):
        if state.common_epoch and should_stop_group(state.common_epoch, state.stale_epochs,
                                                    maximum_epochs, minimum_epochs, patience):
            break
        started = time.perf_counter()
        records = {seed: track.epoch(data, epoch, batch_size) for seed, track in tracks.items()}
        means = synchronized_epoch_rows([records[s]['validation']['raw']['macro_f1'] for s in SEEDS],
                                        [records[s]['validation']['ema']['macro_f1'] for s in SEEDS])
        mark_epoch(state, epoch, records, *means)
        save_checkpoint(directory, dict(identity=identity, state=state.to_dict(),
                        tracks={s: t.export() for s, t in tracks.items()}))
        progress = dict(common_epoch=epoch, until_epoch=until_epoch, arm=arm,
                        protocol=identity['protocol'], fold=identity['fold'],
                        three_seed_TRAIN_loss=[records[s]['train_loss'] for s in SEEDS],
                        mean_raw_macro_f1=means[0], mean_EMA_macro_f1=means[1],
                        epoch_seconds=time.perf_counter()-started,
                        updated_at_unix=time.time(), Test_allowed=False)
        temporary = Path(directory) / 'PROGRESS.tmp'
        temporary.write_text(json.dumps(progress, indent=2))
        temporary.replace(Path(directory) / 'PROGRESS.json')
        print('COMMON_EPOCH', json.dumps(progress), flush=True)
    return dict(identity=identity, state=state.to_dict(), tracks={s: t.export() for s, t in tracks.items()})
