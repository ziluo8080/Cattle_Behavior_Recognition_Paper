"""Cloud-only fixed four-arm clean development experiment, without TEST access."""
import json
import hashlib
import time
from pathlib import Path
START = time.monotonic()
START_WALL = time.time()
import numpy as np
import pandas as pd
import torch
from torch import nn
from scripts.sensor_local_tcn_model import ResidualSensor
from scripts.sensor_reliability_data import DamagePool

CONFIG = Path('configs/sensor_local_tcn_residual_20261009.json')
ROOT = Path('/q3/mmcows-sensor-behavior-20261007')
NAMES = ['行走', '站立', '抬头采食', '低头采食', '舔舐', '饮水', '躺卧']


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def f1(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    den = cm.sum(-1) + cm.sum(-2)
    return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0)


def metrics(y, pred):
    if not np.isin(y, range(1, 8)).all() or not np.isin(pred, range(1, 8)).all():
        raise AssertionError('Original seven labels only')
    cm = np.bincount((y - 1) * 7 + pred - 1, minlength=49).reshape(7, 7)
    tp = cm.diagonal()
    precision = np.divide(tp, cm.sum(0), out=np.zeros(7), where=cm.sum(0) > 0)
    recall = np.divide(tp, cm.sum(1), out=np.zeros(7), where=cm.sum(1) > 0)
    return dict(macro_f1=float(f1(cm).mean()), f1=f1(cm).tolist(), precision=precision.tolist(),
                recall=recall.tolist(), support=cm.sum(1).tolist(), CM=cm.tolist())


def budget(cfg, stage):
    elapsed = time.monotonic() - START
    if elapsed > cfg['maximum_wall_seconds']:
        write(Path(cfg['output']) / 'PARTIAL.json', dict(stage=stage, seconds=elapsed, complete=False))
        raise RuntimeError('Fixed budget exceeded; no reset')
    return elapsed


def provenance(cfg):
    parent = read(cfg['parent_configuration'])
    paths = [CONFIG, Path(__file__), Path('scripts/sensor_local_tcn_model.py'),
             Path('scripts/run_sensor_local_tcn_residual.sh'), Path(cfg['parent_configuration'])]
    paths += list(Path('q3_sensor').glob('*.py'))
    record = {str(p): sha(p) for p in paths}
    for protocol in cfg['protocols']:
        cache = Path(parent['cache']) / protocol
        receipt = read(cache / 'COMPLETE.json')
        record[str(cache / 'COMPLETE.json')] = sha(cache / 'COMPLETE.json')
        for path, expected in receipt['identity']['sources'].items():
            if sha(path) != expected:
                raise AssertionError('Frozen parent scientific source changed: ' + path)
            record[path] = expected
        for name, key in [('targets.jsonl', 'target_sha256'), ('normalizer.json', 'normalizer_sha256'), ('fit_statistics.json', 'statistics_sha256')]:
            path = cache / name
            if sha(path) != receipt[key]:
                raise AssertionError('Frozen metadata changed')
            record[str(path)] = receipt[key]
        for role in ['train', 'val']:
            for modality in ['uwb', 'hd', 'akl']:
                for field in ['x', 'valid', 'fresh', 'age']:
                    name = f'{role}/{modality}_clean_{field}.npy'
                    if sha(cache / name) != receipt['arrays'][name]:
                        raise AssertionError('Frozen clean array changed')
                    record[str(cache / name)] = receipt['arrays'][name]
        for seed in cfg['seeds']:
            directory = Path(parent['output']) / protocol / f'P0_seed{seed}'
            done = read(directory / 'COMPLETE.json')
            for name, key in [('initial_state.pt', 'initial_state_sha256'), ('model.pt', 'model_sha256')]:
                if sha(directory / name) != done[key]:
                    raise AssertionError('Frozen P0 initialization/model changed')
                record[str(directory / name)] = done[key]
            for name in ['COMPLETE.json', 'clean_predictions.csv']:
                record[str(directory / name)] = sha(directory / name)
        budget(cfg, 'provenance_' + protocol)
    return parent, record


def dataset(parent, protocol):
    cache = Path(parent['cache']) / protocol
    targets = pd.read_json(cache / 'targets.jsonl', lines=True, convert_dates=False)
    if set(targets.split) != {'train', 'val'} or targets.duplicated(['cow_id', 'timestamp']).any():
        raise AssertionError('Only isolated development roles allowed')
    train = targets[targets.split.eq('train')].reset_index(drop=True)
    val = targets[targets.split.eq('val')].reset_index(drop=True)
    if protocol == 'OS' and (set(train.cow_id) != set(range(1, 7)) or set(val.cow_id) != {7, 8}):
        raise AssertionError('OS roles changed')
    stats = read(cache / 'fit_statistics.json')
    counts = np.bincount(train.label.to_numpy(), minlength=8)[1:8]
    np.testing.assert_array_equal(counts, stats['class_counts'])
    np.testing.assert_allclose(stats['prior'], counts / counts.sum(), atol=1e-15, rtol=0)
    np.testing.assert_allclose(stats['CE_weights'], np.sqrt(counts.sum() / (7 * counts)), atol=1e-15, rtol=0)
    return train, val, stats, DamagePool(cache / 'train', ['clean']), DamagePool(cache / 'val', ['clean'])


def initial(parent, protocol, seed):
    return torch.load(Path(parent['output']) / protocol / f'P0_seed{seed}' / 'initial_state.pt', map_location='cpu', weights_only=True)


def make(cfg, parent, protocol, arm, seed, stats):
    model = ResidualSensor(arm, stats['prior'], initial(parent, protocol, seed), seed + cfg['training']['branch_seed_offset'])
    return model.cuda()


def sanity(cfg, parent):
    train, _, stats, pool, _ = dataset(parent, 'TS')
    batch = pool.batch(np.arange(32))
    y = torch.as_tensor(train.label.to_numpy()[:32] - 1, dtype=torch.long, device='cuda')
    weights = torch.tensor(stats['CE_weights'], dtype=torch.float32, device='cuda')
    criterion = nn.CrossEntropyLoss(weight=weights)
    counts, witness, reference, update_states = {}, {}, None, {}
    branch_states = {}
    rng_witness = {}
    for arm in cfg['arms']:
        model = make(cfg, parent, 'TS', arm, 17, stats)
        counts[arm] = sum(p.numel() for p in model.parameters())
        model.eval()
        with torch.no_grad():
            logits = model(batch)
        if reference is None:
            reference = logits.clone()
        torch.testing.assert_close(logits, reference, atol=1e-6, rtol=0)
        if arm in ['StaticLocal', 'LocalTCN']:
            branch_states[arm] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if not k.startswith('base.')}
        empty = {m: {k: v.clone() for k, v in d.items()} for m, d in batch.items()}
        for d in empty.values():
            d['valid'].zero_()
        with torch.no_grad():
            torch.testing.assert_close(model(empty), model.base.prior_logits[None].expand(32, -1), atol=0, rtol=0)
        model.train()
        torch.manual_seed(17 + cfg['training']['dropout_seed_offset'])
        torch.cuda.manual_seed_all(17 + cfg['training']['dropout_seed_offset'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['learning_rate'], weight_decay=cfg['training']['weight_decay'])
        rng_before = torch.cuda.get_rng_state().clone()
        loss = criterion(model(batch), y)
        rng_after = torch.cuda.get_rng_state().clone()
        rng_witness[arm] = (rng_before, rng_after)
        torch.save(dict(before=rng_before, after=rng_after), Path(cfg['output']) / f'sanity_rng_{arm}.pt')
        loss.backward()
        last_gradient_norm = None
        if arm != 'Base':
            last_gradient_norm = float(model.residual[-1].weight.grad.norm())
            if not np.isfinite(last_gradient_norm) or last_gradient_norm <= 0:
                raise AssertionError('Residual output layer must receive nonzero first gradient')
        optimizer.step()
        update_states[arm] = {k: v.detach().cpu().clone() for k, v in model.base.state_dict().items()}
        torch.save(update_states[arm], Path(cfg['output']) / f'sanity_first_update_{arm}.pt')
        witness[arm] = dict(loss=float(loss.detach()), residual_first_gradient_norm=last_gradient_norm,
            base_updated_first_step={k: v.detach().cpu().numpy().tolist()
            for k, v in list(model.base.state_dict().items()) if k in ['head.3.bias']})
        if arm != 'Base':
            optimizer.zero_grad(set_to_none=True)
            criterion(model(batch), y).backward()
            if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
                raise AssertionError('Missing or nonfinite gradient after initial residual update')
            hidden_norm = float(model.residual[0].weight.grad.norm())
            local_norms = {m: float(sum(p.grad.square().sum() for p in block.parameters()).sqrt())
                           for m, block in model.local.items()} if arm in ['StaticLocal', 'LocalTCN'] else {}
            if hidden_norm <= 0 or any(value <= 0 for value in local_norms.values()):
                raise AssertionError('Residual hidden/conv parameters must get nonzero second gradients')
            witness[arm]['second_hidden_gradient_norm'] = hidden_norm
            witness[arm]['second_local_gradient_norms'] = local_norms
        del model
    assert counts['Base'] == 86695
    assert (counts['LocalTCN'] - counts['Base']) / counts['Base'] <= cfg['parameter_budget']['maximum_candidate_increase_fraction']
    assert counts['StaticLocal'] == counts['LocalTCN']
    assert abs(counts['DenseResidual'] - counts['LocalTCN']) / counts['LocalTCN'] <= cfg['parameter_budget']['capacity_control_tolerance_fraction']
    for k, v in branch_states['StaticLocal'].items():
        torch.testing.assert_close(v, branch_states['LocalTCN'][k], atol=0, rtol=0)
    for arm in cfg['arms']:
        for state, base_state in zip(rng_witness[arm], rng_witness['Base']):
            torch.testing.assert_close(state, base_state, atol=0, rtol=0)
        np.testing.assert_allclose(witness[arm]['loss'], witness['Base']['loss'], atol=1e-6, rtol=0)
        np.testing.assert_allclose(witness[arm]['base_updated_first_step']['head.3.bias'], witness['Base']['base_updated_first_step']['head.3.bias'], atol=1e-6, rtol=0)
        for key, value in update_states[arm].items():
            torch.testing.assert_close(value, update_states['Base'][key], atol=1e-6, rtol=0)
    write(Path(cfg['output']) / 'SANITY.json', dict(status='PASS', parameters=counts, first_batch=witness,
        sanity_optimizer_steps=4, real_TRAIN_targets=32, initial_logits_match=True,
        StaticLocal_LocalTCN_initial_branch_equal=True, all_missing_prior_checked=True))
    print('LOCAL_TCN_SANITY_PASS', counts, flush=True)


def train_one(cfg, parent, protocol, arm, seed):
    directory = Path(cfg['output']) / protocol / f'{arm}_seed{seed}'
    if directory.exists():
        raise RuntimeError('No fit retry or budget reset')
    directory.mkdir(parents=True)
    write(directory / 'STARTED.json', dict(protocol=protocol, arm=arm, seed=seed, budget_elapsed=budget(cfg, 'start_fit')))
    train, val, stats, train_pool, val_pool = dataset(parent, protocol)
    model = make(cfg, parent, protocol, arm, seed, stats)
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, directory / 'initial_state.pt')
    t = cfg['training']
    torch.manual_seed(seed + t['dropout_seed_offset'])
    torch.cuda.manual_seed_all(seed + t['dropout_seed_offset'])
    rng = np.random.default_rng(seed)
    y = torch.as_tensor(train.label.to_numpy() - 1, dtype=torch.long, device='cuda')
    weights = torch.as_tensor(stats['CE_weights'], dtype=torch.float32, device='cuda')
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t['learning_rate'], weight_decay=t['weight_decay'])
    history, order_hashes = [], []
    torch.cuda.reset_peak_memory_stats()
    fit_start = time.monotonic()
    for epoch in range(1, t['epochs'] + 1):
        epoch_start = time.monotonic()
        model.train()
        order = rng.permutation(len(train))
        order_hashes.append(hashlib.sha256(order.astype('<i8').tobytes()).hexdigest())
        numerator = denominator = 0.
        for at in range(0, len(order), t['batch_size']):
            idx = order[at:at + t['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(train_pool.batch(idx)), y[idx])
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            weight_sum = float(weights[y[idx]].sum())
            numerator += float(loss.detach()) * weight_sum
            denominator += weight_sum
        history.append(dict(epoch=epoch, train_CE=numerator / denominator,
                            epoch_seconds=time.monotonic() - epoch_start, elapsed_seconds=time.monotonic() - fit_start))
        write(directory / 'history.json', history)
        budget(cfg, 'epoch')
        print('LOCAL_TCN_EPOCH', protocol, arm, seed, epoch, history[-1], flush=True)
        pilot = cfg['pilot']
        if (protocol, arm, seed, epoch) == (pilot['protocol'], pilot['arm'], pilot['seed'], pilot['epochs']):
            per_epoch = max(row['epoch_seconds'] for row in history[1:])
            forecast = budget(cfg, 'pilot_5_epochs') + pilot['remaining_epochs'] * per_epoch * pilot['factor'] + pilot['summary_and_other_overhead_allowance_seconds']
            write(Path(cfg['output']) / 'PILOT.json', dict(pilot_first_fit_prefix=True,
                completed_epochs=epoch, per_epoch=per_epoch, forecast=forecast,
                remaining_admitted=forecast <= cfg['maximum_wall_seconds']))
            if forecast > cfg['maximum_wall_seconds']:
                torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    kind=arm, completed_epochs=epoch, fit='TRAIN_only', resumable=False), directory / 'partial_model.pt')
                raise RuntimeError('Budget insufficient for remaining frozen queue; not a method failure')
    torch.cuda.synchronize()
    training_seconds = time.monotonic() - fit_start
    model.eval()
    logits = []
    with torch.no_grad():
        for at in range(0, len(val), t['batch_size']):
            logits.append(model(val_pool.batch(np.arange(at, min(at + t['batch_size'], len(val))))).cpu().numpy())
    logits = np.concatenate(logits)
    if not np.isfinite(logits).all():
        raise RuntimeError('Nonfinite final logits')
    pred = logits.argmax(1) + 1
    table = val[['cow_id', 'timestamp', 'label']].copy()
    table['prediction'] = pred
    table.to_csv(directory / 'predictions.csv', index=False)
    np.save(directory / 'val_logits.npy', logits)
    torch.save(dict(state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, prior=stats['prior'],
                    kind=arm, seed=seed, protocol=protocol, fit='TRAIN_only', selection='fixed_final_epoch',
                    target_sha256=sha(Path(parent['cache']) / protocol / 'targets.jsonl')), directory / 'model.pt')
    record = dict(protocol=protocol, arm=arm, seed=seed, epochs=t['epochs'], train_targets=len(train), val_targets=len(val),
        parameters=sum(p.numel() for p in model.parameters()), scores=metrics(val.label.to_numpy(), pred),
        training_seconds=training_seconds, total_seconds=time.monotonic() - fit_start,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(), order_hashes=order_hashes,
        files={name: sha(directory / name) for name in ['model.pt', 'initial_state.pt', 'history.json', 'predictions.csv', 'val_logits.npy']})
    if arm == 'Base':
        old = pd.read_csv(Path(parent['output']) / protocol / f'P0_seed{seed}' / 'clean_predictions.csv')
        pd.testing.assert_frame_equal(old[['cow_id', 'timestamp', 'label']], val[['cow_id', 'timestamp', 'label']])
        record['matched_old_P0_predictions'] = bool(np.array_equal(pred, old.prediction.to_numpy()))
        if not record['matched_old_P0_predictions']:
            raise AssertionError('Matched base differs from original P0; preserve failure')
    write(directory / 'COMPLETE.json', record)
    print('LOCAL_TCN_FIT_COMPLETE', protocol, arm, seed, record['scores']['macro_f1'], flush=True)
    del model
    torch.cuda.empty_cache()
    return record


def summarize(cfg, parent):
    root = Path(cfg['output'])
    result, group_scores, flows = dict(protocols={}, TEST_scoring=False, audit_status='pending'), [], []
    for protocol in cfg['protocols']:
        targets = pd.read_json(Path(parent['cache']) / protocol / 'targets.jsonl', lines=True, convert_dates=False)
        val = targets[targets.split.eq('val')].reset_index(drop=True)
        y = val.label.to_numpy()
        _, blocks = np.unique(val.timestamp.to_numpy() // cfg['interval']['block_seconds'], return_inverse=True)
        nb = int(blocks.max()) + 1
        draws = np.random.default_rng(cfg['interval']['seed']).multinomial(nb, np.full(nb, 1 / nb), size=cfg['interval']['replicates'])
        scores, boots, predictions = {}, {}, {}
        for arm in cfg['arms']:
            per_seed, boot, classes, resources = [], [], [], []
            predictions[arm] = []
            for seed in cfg['seeds']:
                directory = root / protocol / f'{arm}_seed{seed}'
                done = read(directory / 'COMPLETE.json')
                for name, expected in done['files'].items():
                    if sha(directory / name) != expected:
                        raise AssertionError('Completed fit file changed')
                table = pd.read_csv(directory / 'predictions.csv')
                pd.testing.assert_frame_equal(table[['cow_id', 'timestamp', 'label']], val[['cow_id', 'timestamp', 'label']])
                pred = table.prediction.to_numpy()
                predictions[arm].append(pred)
                s = metrics(y, pred)
                per_seed.append(s['macro_f1']); classes.append(s['f1']); resources.append(done)
                cm = np.bincount(blocks * 49 + (y - 1) * 7 + pred - 1, minlength=nb * 49).reshape(nb, 7, 7)
                boot.append(f1(np.einsum('rb,bjk->rjk', draws, cm)).mean(1))
                masks = {f'cow_{cow}': val.cow_id.to_numpy() == cow for cow in sorted(val.cow_id.unique())}
                masks.update(history_prefix_short=val.timestamp.to_numpy() - val.start.to_numpy() < 60,
                             history_full=val.timestamp.to_numpy() - val.start.to_numpy() >= 60)
                for name, mask in masks.items():
                    group_scores.append(dict(protocol=protocol, arm=arm, seed=seed, group=name, targets=int(mask.sum()), scores=metrics(y[mask], pred[mask])))
            scores[arm] = dict(mean=float(np.mean(per_seed)), per_seed=per_seed, seed_std_population=float(np.std(per_seed)),
                f1=np.mean(classes, 0).tolist(), resources=resources)
            boots[arm] = np.mean(boot, 0)
        pairs = {}
        for control in ['Base', 'DenseResidual', 'StaticLocal']:
            d = scores['LocalTCN']['mean'] - scores[control]['mean']
            pairs[control] = dict(delta=d, relative_gain=d / scores[control]['mean'],
                CI95=np.quantile(boots['LocalTCN'] - boots[control], [.025, .975]).tolist(),
                class_delta=(np.asarray(scores['LocalTCN']['f1']) - scores[control]['f1']).tolist(),
                per_seed_delta=(np.asarray(scores['LocalTCN']['per_seed']) - scores[control]['per_seed']).tolist())
            for i, seed in enumerate(cfg['seeds']):
                p, q = predictions[control][i], predictions['LocalTCN'][i]
                rescue, harm = (p != y) & (q == y), (p == y) & (q != y)
                flows.append(dict(protocol=protocol, seed=seed, control=control,
                    rescued_by_class=np.bincount(y[rescue], minlength=8)[1:8].tolist(),
                    harmed_by_class=np.bincount(y[harm], minlength=8)[1:8].tolist(),
                    changed_flow=np.bincount((y[p != q] - 1) * 49 + (p[p != q] - 1) * 7 + q[p != q] - 1, minlength=343).reshape(7, 7, 7).tolist()))
        g = cfg['gates']
        gates = dict(base_gain=pairs['Base']['delta'] >= g['candidate_macro_gain_vs_Base'],
            classes=min(pairs['Base']['class_delta']) >= -g['class_drop_limit_vs_Base'],
            dense_gain=pairs['DenseResidual']['delta'] >= g['candidate_macro_gain_vs_DenseResidual'],
            static_gain=pairs['StaticLocal']['delta'] >= g['candidate_macro_gain_vs_StaticLocal'],
            positive_CI=all(p['CI95'][0] > 0 for p in pairs.values()))
        gates['seed_consistency'] = sum(d > 0 for d in pairs['Base']['per_seed_delta']) >= g['positive_Base_delta_seed_count']
        gates['passed'] = all(gates.values())
        result['protocols'][protocol] = dict(scores=scores, comparisons=pairs, gates=gates)
    result['adoptable'] = all(p['gates']['passed'] for p in result['protocols'].values())
    result['status'] = 'supported_pending_audit' if result['adoptable'] else 'method_failed_frozen_pending_audit'
    write(root / 'verified_summary.json', result)
    write(root / 'groups.json', group_scores); write(root / 'error_flows.json', flows)
    lines = ['# 局部TCN残差模块四臂开发验证', '', '全部同输入/sqrt/20epoch最终模型，0TEST评分；方法成绩待独立核验。', '']
    for protocol, p in result['protocols'].items():
        lines += [f'## {protocol}', '', '| 方法 | macroF1% | 三seed% | 参数 |', '|---|---:|---|---:|']
        for arm, s in p['scores'].items():
            lines.append(f"| {arm} | {s['mean']*100:.6f} | "+'/'.join(f'{x*100:.6f}' for x in s['per_seed'])+f" | {s['resources'][0]['parameters']} |")
        lines += ['', '| 对照 | 候选增益pp | 95%区间pp |', '|---|---:|---|']
        for c, d in p['comparisons'].items():
            lines.append(f"| {c} | {d['delta']*100:+.6f} | [{d['CI95'][0]*100:+.6f},{d['CI95'][1]*100:+.6f}] |")
        lines += ['', '| 类别 | Base% | Dense% | Static% | LocalTCN% | Local−Base pp |', '|---|---:|---:|---:|---:|---:|']
        for i, name in enumerate(NAMES):
            lines.append('| '+name+' | '+' | '.join(f"{p['scores'][a]['f1'][i]*100:.6f}" for a in cfg['arms'])+f" | {p['comparisons']['Base']['class_delta'][i]*100:+.6f} |")
        lines += ['', '固定采用门：'+json.dumps(p['gates'], ensure_ascii=False), '']
    lines += [result['status'], '', '旧TS首筛与此原生开发名单不同，不能比较绝对分。已有曝光、OS VAL两牛、固定模型条件区间；无原创/新场地/故障鲁棒性主张。控制胜出不改名为LocalTCN成功。']
    (root / 'RESULTS.md').write_text('\n'.join(lines), encoding='utf-8')
    return result


def main():
    if Path.cwd() != ROOT or not torch.cuda.is_available():
        raise RuntimeError('bitahub-root CUDA-only scientific execution')
    cfg = read(CONFIG)
    root = Path(cfg['output'])
    if root.exists():
        raise RuntimeError('Preserve existing attempt; no automatic restart')
    root.mkdir(parents=True)
    write(root / 'BUDGET_START.json', dict(unix=START_WALL, limit=cfg['maximum_wall_seconds']))
    torch.set_num_threads(cfg['training']['threads'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        parent, identity = provenance(cfg)
        write(root / 'IDENTITY.json', dict(inputs=identity, torch=torch.__version__, GPU=torch.cuda.get_device_name(0)))
        sanity(cfg, parent)
        pilot = cfg['pilot']
        train_one(cfg, parent, pilot['protocol'], pilot['arm'], pilot['seed'])
        for protocol in cfg['protocols']:
            for seed in cfg['seeds']:
                for arm in cfg['arms']:
                    if (protocol, arm, seed) != (pilot['protocol'], pilot['arm'], pilot['seed']):
                        train_one(cfg, parent, protocol, arm, seed)
        result = summarize(cfg, parent)
        for path, expected in identity.items():
            if sha(path) != expected:
                raise AssertionError('Frozen source changed during experiment')
        write(root / 'COMPLETE.json', dict(new_fits=24, trained_epochs=480, sanity_optimizer_steps=4,
            seconds=budget(cfg, 'complete'), TEST_scoring=False, status=result['status'], source_pre_post_equal=True))
        print('LOCAL_TCN_EXPERIMENT_COMPLETE', result['status'], flush=True)
    except Exception as error:
        reason = 'budget_insufficient' if 'Budget insufficient' in str(error) else ('budget_exceeded' if 'Fixed budget exceeded' in str(error) else 'execution_error')
        write(root / 'FAILURE.json', dict(error=repr(error), reason_code=reason,
            method_effectiveness_established=False, seconds=time.monotonic() - START, complete=False))
        raise


if __name__ == '__main__':
    main()
