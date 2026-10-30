"""One-change, TRAIN-internal replication; original sealed code stays untouched."""
import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from scripts import run_convtran_context_20261022 as core
from scripts.run_convtran_endpoint_20261022 import model_for, optimizer_for, equal_tree
from scripts.sensor_learned_fusion_data_20261023 import read, save, sha, make_splits
from scripts.sensor_uwb_event_motion_20261020 import ParameterEMA
from scripts.sensor_xgb_b_fusion_20261022 import validate_probabilities, temper, fuse

CONFIG = Path('configs/sensor_zero_head_replication_20261024_v1.json')
MODULE = 'scripts.run_zero_head_replication_20261024'
FIELDS = ('dynamic', 'state', 'state_valid', 'context', 'current')


def configure(cuda=False):
    assert Path.cwd() == core.ROOT, 'Scientific workloads are cloud-only'
    assert (torch.__version__, np.__version__) == ('2.4.1+cu121', '1.24.4')
    if cuda:
        core.configure()
    else:
        torch.set_num_threads(4)
        torch.use_deterministic_algorithms(True)
    cfg = read(CONFIG)
    assert cfg['protocols'] == ['TS', 'OS'] and cfg['inners'] == [0, 1, 2]
    assert cfg['seeds'] == [17, 23, 29] and cfg['new_fits'] == 18 and cfg['epochs'] == 20
    assert all(cfg[k] is False for k in ('Test_allowed', 'outer_VAL_allowed', 'formal_refit_allowed', 'automatic_expansion', 'hyperparameter_search'))
    return cfg


def implementation(cfg):
    paths = {CONFIG, Path(cfg['plan']), Path(cfg['environment_spec']), Path(cfg['B_config']),
             Path('.aris/compute/bitahub-root-zero-head-20261024.md'), Path(__file__).relative_to(core.ROOT) if Path(__file__).is_absolute() else Path(__file__)}
    paths.add(Path('scripts/report_zero_head_replication_20261024.py'))
    paths.add(Path('scripts/run_convtran_endpoint_20261022.py'))
    old = read('reports/sensor_tree_anchored_dynamic_residual_20261024_v1/METHOD_LOCK.json')['code']
    # Include the transitive sealed runtime, including vendor and q3 modules.
    for p, digest in old.items():
        if p.endswith('.py'):
            assert sha(p) == digest, p
            paths.add(Path(p))
    paths.update(Path('third_party/convtran_context_20261022_v1').rglob('*.py'))
    return {p.as_posix(): sha(p) for p in sorted(paths)}


def native(cfg, prior, zero=True, device='cuda'):
    model = model_for('EndpointResidual_B', read(cfg['B_config']), prior).to(device)
    assert sum(p.numel() for p in model.parameters()) == 213119
    if zero:
        nn.init.zeros_(model.head[-1].weight)
        nn.init.zeros_(model.head[-1].bias)
    return model


def init_match(cfg, prior, seed, device):
    core.seed_all(seed)
    original = native(cfg, prior, False, device)
    expected_rng = core.rng()
    core.seed_all(seed)
    zero = native(cfg, prior, True, device)
    equal_tree(expected_rng, core.rng())
    for key, value in original.state_dict().items():
        if key in ('head.3.weight', 'head.3.bias'):
            assert torch.count_nonzero(zero.state_dict()[key]) == 0
        else:
            equal_tree(value, zero.state_dict()[key])
    return zero


def minibatch(data, ids):
    device = data['tensors']['dynamic'].device
    index = torch.as_tensor(ids, device=device)
    return {k: v[index] for k, v in data['tensors'].items() if k != 'labels'}


def update(model, opt, ema, data, ids):
    if data['tensors']['dynamic'].is_cuda:
        return core.update(model, opt, ema, data, ids)
    # The documented CPU witness cannot call the legacy CUDA-only minibatcher.
    model.train()
    opt.zero_grad(set_to_none=True)
    labels = data['tensors']['labels'][torch.as_tensor(ids)]
    loss = torch.nn.functional.cross_entropy(model(minibatch(data, ids)), labels, weight=data['weights'])
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    ema.update(model)
    return float(loss.detach()), float(data['weights'][labels].sum())


@torch.no_grad()
def infer(model, data, ids, batch_size=256):
    model.eval()
    parts = []
    for start in range(0, len(ids), batch_size):
        logits = model(minibatch(data, ids[start:start + batch_size]))
        assert torch.isfinite(logits).all()
        parts.append(logits.double().softmax(-1).cpu().numpy())
    return validate_probabilities(np.concatenate(parts))


def checkpoint(path, model, opt, ema, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), ema=ema.model.state_dict(),
                    ema_updates=ema.updates, rng=core.rng(), **metadata), temporary)
    temporary.replace(path)


def restore_check(cfg, data, path, device):
    bcfg = read(cfg['B_config'])
    core.seed_all(17)
    model = native(cfg, data['prior'], True, device)
    opt = optimizer_for(model, bcfg)
    ema = ParameterEMA(model, bcfg['ema_decay'], bcfg['ema_warmup_steps'])
    ids = data['train'][:67]
    update(model, opt, ema, data, ids)
    checkpoint(path, model, opt, ema, epoch=0)
    expected = update(model, opt, ema, data, ids)
    expected_rng = core.rng()
    payload = torch.load(path, map_location='cpu', weights_only=False)
    other = native(cfg, data['prior'], True, device)
    other.load_state_dict(payload['model'])
    oo = optimizer_for(other, bcfg)
    oo.load_state_dict(payload['optimizer'])
    ee = ParameterEMA(other, bcfg['ema_decay'], bcfg['ema_warmup_steps'])
    ee.model.load_state_dict(payload['ema'])
    ee.updates = payload['ema_updates']
    core.restore_rng(payload['rng'])
    equal_tree(expected, update(other, oo, ee, data, ids))
    equal_tree(expected_rng, core.rng())
    for a, b in ((model.state_dict(), other.state_dict()), (opt.state_dict(), oo.state_dict()), (ema.model.state_dict(), ee.model.state_dict())):
        equal_tree(a, b)
    for a, b in ((model, other), (ema.model, ee.model)):
        equal_tree(infer(a, data, ids), infer(b, data, ids))
    return dict(model_optimizer_EMA_RNG_next_update_exact=True, raw_EMA_tail67_exact=True)


def witness(cfg, output, cpu):
    device = 'cpu' if cpu else 'cuda'
    lock = None if cpu else core.acquire_gpu()
    core.seed_all(17)
    n = 67
    dynamic = torch.randn(n, 100, 8, device=device)
    dynamic[:, :, 6] = 1
    dynamic[:, :, 7] = .1
    state = torch.randn(n, 60, 26, device=device)
    prior = np.full(7, 1 / 7)
    data = dict(prior=prior, train=np.arange(n), weights=torch.ones(7, device=device),
                tensors=dict(dynamic=dynamic, state=state, state_valid=torch.ones(n, 60, device=device, dtype=torch.bool),
                             context=torch.randn(n, 136, device=device), current=state[:, -1], labels=torch.arange(n, device=device) % 7))
    model = init_match(cfg, prior, 17, device)
    opt = optimizer_for(model, read(cfg['B_config']))
    ema = ParameterEMA(model, .998, 200)
    update(model, opt, ema, data, data['train'])
    assert model.dynamic.embedding[0].weight.grad.abs().sum() == 0
    assert model.head[-1].weight.grad.abs().sum() > 0
    update(model, opt, ema, data, data['train'])
    assert model.dynamic.embedding[0].weight.grad.abs().sum() > 0
    missing = {k: v[:3].clone() for k, v in data['tensors'].items() if k != 'labels'}
    missing['dynamic'][:, :, 6] = 0
    missing['state_valid'][:] = False
    model.eval()
    equal_tree(model(missing), model.prior_logits[None].expand(3, -1))
    record = restore_check(cfg, data, Path(output).with_suffix('.pt'), device)
    save(output, dict(status='PASS', CUDA_validated=not cpu, real_data_rows_read=0,
                      parameters=213119, initialization_and_RNG_matched=True,
                      first_step_backbone_gradient_zero=True, second_step_backbone_gradient_nonzero=True,
                      missing_fallback_exact=True, restore=record, implementation_files=implementation(cfg)))
    if lock:
        lock.close()
    print('ZERO_HEAD_WITNESS_PASS', flush=True)


def load_data(cfg, protocol, inner):
    directory = Path(cfg['output']) / 'data' / protocol / f'inner{inner}'
    receipt = read(directory / 'RECEIPT.json')
    for name, digest in receipt['files'].items():
        assert sha(directory / name) == digest, name
    with np.load(directory / 'FIT.npz', allow_pickle=False) as fit, np.load(directory / 'DEV_INPUTS.npz', allow_pickle=False) as held:
        assert 'labels' in fit.files and 'labels' not in held.files
        nfit = len(fit['labels'])
        values = {k: np.concatenate((fit[k], held[k]), axis=0) for k in FIELDS}
        labels = np.concatenate((fit['labels'], np.full(len(held['dynamic']), -100, np.int64)))
        counts = np.bincount(fit['labels'], minlength=7)
    assert (counts > 0).all()
    tensors = {k: torch.as_tensor(v, device='cuda') for k, v in values.items()}
    tensors['labels'] = torch.as_tensor(labels, dtype=torch.long, device='cuda')
    return dict(prior=counts / counts.sum(), train=np.arange(nfit), evaluation=np.arange(nfit, len(labels)),
                weights=torch.tensor(np.sqrt(counts.sum() / (7 * counts)), dtype=torch.float32, device='cuda'), tensors=tensors)


def prepare(cfg, repair=False):
    import json
    import xgboost as xgb
    assert xgb.__version__ == '2.1.4'
    out = Path(cfg['output'])
    if repair:
        assert (out / 'PREPARE_STARTED.json').exists()
        assert not any((out / n).exists() for n in ['PREPARED.json', 'SEAL.json'])
        marker = 'PREPARE_REPAIR_STARTED.json' if not (out / 'PREPARE_REPAIR_STARTED.json').exists() else 'PREPARE_REPAIR2_STARTED.json'
        assert not (out / marker).exists(), 'Preserve bounded explicit repair history'
        save(out / marker, dict(pid=os.getpid(), unix=time.time(),
             reason='Match original XGB normalization and frozen EMA inference metadata; original diagnostics retained; no scientific fits started'))
    else:
        assert not (out / 'PREPARE_STARTED.json').exists()
        save(out / 'PREPARE_STARTED.json', dict(pid=os.getpid(), unix=time.time()))
    lock = core.acquire_gpu()
    parent_seal = read(cfg['parent_seal'])
    old_lock = 'reports/sensor_tree_anchored_dynamic_residual_20261024_v1/METHOD_LOCK.json'
    frozen = {cfg['parent_seal']: sha(cfg['parent_seal']), old_lock: sha(old_lock)}
    records = []
    for protocol in cfg['protocols']:
        roster = Path(cfg['source']) / 'data' / protocol / 'targets.jsonl'
        assert sha(roster) == parent_seal['files'][str(roster)]
        frozen[str(roster)] = sha(roster)
        rows = [json.loads(line) for line in roster.read_text().splitlines()]
        splits = make_splits(rows, protocol, 3)
        for inner in cfg['inners']:
            cache = Path(cfg['parent_cache']) / protocol / f'inner{inner}'
            expert = Path(cfg['parent_experts']) / protocol / f'inner{inner}'
            receipt_path = cache / 'DATA_RECEIPT.json'
            assert sha(receipt_path) == parent_seal['files'][str(receipt_path)]
            frozen[str(receipt_path)] = sha(receipt_path)
            receipt = read(receipt_path)
            for name, digest in receipt['outputs'].items():
                path = cache / name
                assert sha(path) == digest
                frozen[str(path)] = digest
            for name, record in receipt['source_files'].items():
                assert sha(name) == record['sha256']
                frozen[name] = record['sha256']
            with np.load(cache / 'DATA.npz', allow_pickle=False) as z:
                data = {k: z[k] for k in z.files}
            split = splits[inner]
            for key in ('train', 'held', 'original_ids'):
                np.testing.assert_array_equal(data[key], split[key])
            np.testing.assert_array_equal(data['labels'], [r['label'] - 1 for r in split['rows']])
            fit, held = data['train'], data['held']
            assert not np.intersect1d(fit, held).size
            np.testing.assert_array_equal(data['original_ids'][held], np.load(expert / 'held_original_ids.npy'))
            np.testing.assert_array_equal(data['labels'][held] + 1, np.load(expert / 'held_truth.npy'))
            d = out / 'data' / protocol / f'inner{inner}'
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / 'FIT.npz', **{k: data[k][fit] for k in FIELDS}, labels=data['labels'][fit])
            np.savez(d / 'DEV_INPUTS.npz', **{k: data[k][held] for k in FIELDS})
            gold = [dict(label=int(data['labels'][i]) + 1, original_id=int(data['original_ids'][i])) for i in held]
            (d / 'DEV_TARGETS.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in gold), encoding='utf-8')
            save(d / 'RECEIPT.json', dict(protocol=protocol, inner=inner, fit_rows=len(fit), held_rows=len(held),
                 fit_counts=np.bincount(data['labels'][fit], minlength=7).tolist(), held_counts=np.bincount(data['labels'][held], minlength=7).tolist(),
                 files={n: sha(d / n) for n in ['FIT.npz', 'DEV_INPUTS.npz', 'DEV_TARGETS.jsonl']}))
            for n in ['FIT.npz', 'DEV_INPUTS.npz', 'DEV_TARGETS.jsonl', 'RECEIPT.json']:
                frozen[str(d / n)] = sha(d / n)
            gd = load_data(cfg, protocol, inner)
            baseline = out / 'baseline' / protocol / f'inner{inner}'
            baseline.mkdir(parents=True, exist_ok=True)
            booster = xgb.Booster()
            booster.load_model(expert / 'XGB_seed17.json')
            booster.set_param({'nthread': 4})
            assert booster.num_boosted_rounds() == 500
            px = booster.predict(xgb.DMatrix(data['features'][held], nthread=4)).astype(np.float64)
            missing = ~((data['dynamic'][held, :, 6] > .5).any(1) | data['state_valid'][held].any(1))
            px[missing] = gd['prior']
            px = validate_probabilities(px)
            np.testing.assert_array_equal(px, np.load(expert / 'XGB_probabilities.npy'))
            np.save(baseline / 'XGB.npy', px)
            for seed in cfg['seeds']:
                init_match(cfg, gd['prior'], seed, 'cuda')
                payload = torch.load(expert / f'B_seed{seed}_latest.pt', map_location='cpu', weights_only=False)
                assert (payload['protocol'], payload['inner'], payload['seed'], payload['epoch']) == (protocol, inner, seed, 20)
                model = native(cfg, gd['prior'], False)
                model.load_state_dict(payload['ema'])
                model.eval().requires_grad_(False)
                probability = infer(model, gd, gd['evaluation'])
                np.testing.assert_array_equal(probability, np.load(expert / f'B_ema_{seed}_probabilities.npy'))
                np.save(baseline / f'B_{seed}.npy', probability)
                np.save(baseline / f'Gate_{seed}.npy', fuse(temper(px, 2), temper(probability, 1), 'margin_select', .7))
                for name in [f'B_seed{seed}_latest.pt', f'B_ema_{seed}_probabilities.npy']:
                    frozen[str(expert / name)] = sha(expert / name)
                del model, payload
            for name in ['XGB_seed17.json', 'XGB_probabilities.npy', 'held_original_ids.npy', 'held_truth.npy', 'COMPLETE.json']:
                frozen[str(expert / name)] = sha(expert / name)
            for path in baseline.iterdir():
                frozen[str(path)] = sha(path)
            records.append(dict(protocol=protocol, inner=inner, XGB_exact=True, B_EMA_seeds_exact=cfg['seeds'], scores_computed=0))
            del gd, data, booster
            gc.collect()
            torch.cuda.empty_cache()
    lock.close()
    save(out / 'PREPARED.json', dict(status='PASS', records=records, files=frozen, new_scientific_fits=0))


def preflight(cfg):
    out = Path(cfg['output'])
    assert not (out / 'PREFLIGHT_STARTED.json').exists()
    save(out / 'PREFLIGHT_STARTED.json', dict(pid=os.getpid(), unix=time.time()))
    assert read(out / 'PREPARED.json')['status'] == 'PASS'
    lock = core.acquire_gpu()
    records = []
    bcfg = read(cfg['B_config'])
    for protocol in cfg['protocols']:
        for inner in cfg['inners']:
            data = load_data(cfg, protocol, inner)
            model = init_match(cfg, data['prior'], 17, 'cuda')
            opt = optimizer_for(model, bcfg)
            ema = ParameterEMA(model, bcfg['ema_decay'], bcfg['ema_warmup_steps'])
            times = []
            for i in range(3):
                tick = time.monotonic()
                core.update(model, opt, ema, data, data['train'][i * 256:(i + 1) * 256])
                torch.cuda.synchronize()
                times.append(time.monotonic() - tick)
            restored = restore_check(cfg, data, out / 'preflight' / f'{protocol}_{inner}.pt', 'cuda')
            estimate = (max(times[1:]) * int(np.ceil(len(data['train']) / 256)) * 20 + 40) * 1.6
            assert estimate < cfg['budget']['job_seconds']
            records.append(dict(protocol=protocol, inner=inner, step_seconds=times, estimated_job_seconds=estimate, restore=restored))
            del data, model, opt, ema
            gc.collect()
            torch.cuda.empty_cache()
    total = sum(r['estimated_job_seconds'] * 3 for r in records) + 180
    assert total < cfg['budget']['total_seconds']
    lock.close()
    save(out / 'PREFLIGHT.json', dict(status='PASS', records=records, estimated_total_seconds=total,
                                    new_held_scores_computed=0, implementation_files=implementation(cfg)))


def seal(cfg):
    out = Path(cfg['output'])
    assert not (out / 'SEAL.json').exists()
    code = implementation(cfg)
    for name in ['RUNTIME_WITNESS.json', 'INDEPENDENT_RUNTIME_WITNESS.json', 'PREFLIGHT.json']:
        record = read(out / name)
        assert record['status'] == 'PASS' and record['implementation_files'] == code, name
    prepared = read(out / 'PREPARED.json')
    for path, digest in prepared['files'].items():
        assert sha(path) == digest, path
    save(out / 'SEAL.json', dict(status='PASS', config=cfg, code=code, files=prepared['files'],
                               evidence={n: sha(out / n) for n in ['PREPARED.json', 'RUNTIME_WITNESS.json', 'INDEPENDENT_RUNTIME_WITNESS.json', 'PREFLIGHT.json']},
                               unix=time.time(), new_candidate_scores_computed=0))


def check(cfg, full=False):
    record = read(Path(cfg['output']) / 'SEAL.json')
    assert record['config'] == cfg and record['code'] == implementation(cfg)
    for path, digest in record['evidence'].items():
        assert sha(Path(cfg['output']) / path) == digest
    if full:
        for path, digest in record['files'].items():
            assert sha(path) == digest, path


def worker(cfg, protocol, inner, seed):
    check(cfg)
    lock = core.acquire_gpu()
    folder = Path(cfg['output']) / 'neural' / protocol / f'inner{inner}' / f'seed{seed}'
    assert not (folder / 'STARTED.json').exists(), 'No automatic retry'
    save(folder / 'STARTED.json', dict(pid=os.getpid(), unix=time.time()))
    started = time.monotonic()
    data = load_data(cfg, protocol, inner)
    bcfg = read(cfg['B_config'])
    core.seed_all(seed)
    model = native(cfg, data['prior'])
    opt = optimizer_for(model, bcfg)
    ema = ParameterEMA(model, bcfg['ema_decay'], bcfg['ema_warmup_steps'])
    history = []
    for epoch in range(1, 21):
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(data['train'])
        tick = time.monotonic()
        numerator = denominator = 0.
        for at in range(0, len(order), bcfg['batch_size']):
            loss, mass = core.update(model, opt, ema, data, order[at:at + bcfg['batch_size']])
            numerator += loss * mass
            denominator += mass
        row = dict(epoch=epoch, train_ce=numerator / denominator, seconds=time.monotonic() - tick)
        assert all(np.isfinite(v) for v in row.values())
        history.append(row)
        save(folder / 'history.json', history)
        checkpoint(folder / 'latest.pt', model, opt, ema, protocol=protocol, inner=inner, seed=seed, epoch=epoch)
        save(folder / 'PROGRESS.json', dict(protocol=protocol, inner=inner, seed=seed, epoch=epoch, updated_unix=time.time(), elapsed_seconds=time.monotonic() - started))
        print(row, flush=True)
    payload = torch.load(folder / 'latest.pt', map_location='cpu', weights_only=False)
    files = {}
    for view, net, key in [('ema', ema.model, 'ema'), ('raw', model, 'model')]:
        probability = infer(net, data, data['evaluation'])
        other = native(cfg, data['prior'])
        other.load_state_dict(payload[key])
        other.eval().requires_grad_(view == 'raw')
        np.testing.assert_array_equal(probability, infer(other, data, data['evaluation']))
        path = folder / f'held_{view}.npy'
        np.save(path, probability)
        files[str(path)] = sha(path)
        del other
    check(cfg)
    files.update({str(folder / n): sha(folder / n) for n in ['latest.pt', 'history.json']})
    save(folder / 'COMPLETE.json', dict(status='COMPLETE', protocol=protocol, inner=inner, seed=seed, epoch=20,
                                      files=files, elapsed_seconds=time.monotonic() - started, held_labels_used=False, replay_exact=True))
    lock.close()


def queue(cfg):
    import signal
    out = Path(cfg['output'])
    check(cfg, True)
    assert not (out / 'LAUNCH.json').exists()
    started = time.monotonic()
    save(out / 'LAUNCH.json', dict(pid=os.getpid(), started_unix=time.time(), new_fits=18, budget=cfg['budget']))
    jobs = [dict(protocol=p, inner=k, seed=s, status='pending') for s in cfg['seeds'] for k in cfg['inners'] for p in cfg['protocols']]
    active = None

    def snapshot(phase):
        save(out / 'PUBLIC_PROGRESS.json', dict(phase=phase, pid=os.getpid(), updated_unix=time.time(), elapsed_seconds=time.monotonic() - started,
             completed=sum(j['status'] == 'completed' for j in jobs), total=18, jobs=jobs))

    def stop():
        if active is not None and active.poll() is None:
            os.killpg(active.pid, signal.SIGTERM)
            try:
                active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(active.pid, signal.SIGKILL)
                active.wait()

    try:
        for job in jobs:
            if time.monotonic() - started > cfg['budget']['total_seconds']:
                raise RuntimeError('Cumulative budget exhausted')
            p, k, s = job['protocol'], job['inner'], job['seed']
            folder = out / 'neural' / p / f'inner{k}' / f'seed{s}'
            log = out / 'logs' / f'{p}_{k}_{s}.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            args = [sys.executable, '-s', '-u', '-m', MODULE, '--mode', 'worker', '--protocol', p, '--inner', str(k), '--seed', str(s)]
            with log.open('wb') as handle:
                active = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                job.update(status='running', pid=active.pid, started_unix=time.time())
                tick = time.monotonic()
                while active.poll() is None:
                    snapshot('FIXED20_TRAINING')
                    if time.monotonic() - tick > cfg['budget']['job_seconds'] or time.monotonic() - started > cfg['budget']['total_seconds']:
                        stop()
                        raise RuntimeError('Scientific budget exhausted; no automatic restart')
                    time.sleep(3)
            assert active.returncode == 0 and read(folder / 'COMPLETE.json')['status'] == 'COMPLETE', str(job)
            job.update(status='completed', finished_unix=time.time())
            snapshot('FIXED20_TRAINING')
        check(cfg, True)
        snapshot('FREEZING_AND_SCORING')
        from scripts.report_zero_head_replication_20261024 import report
        result = report(cfg)
        save(out / 'COMPLETE.json', result)
        snapshot('COMPLETE_STOP_FOR_REVIEW')
    except BaseException:
        stop()
        for j in jobs:
            if j['status'] in ('pending', 'running'):
                j['status'] = 'blocked'
        snapshot('FAILED_STOP_FOR_INSPECTION')
        save(out / 'QUEUE_FAILURE.json', dict(status='FAILED', traceback=traceback.format_exc(), unix=time.time()))
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['cpu_witness', 'witness', 'prepare', 'prepare_repair', 'preflight', 'seal', 'launch', 'queue', 'worker'])
    parser.add_argument('--witness-output')
    parser.add_argument('--protocol', choices=['TS', 'OS'])
    parser.add_argument('--inner', type=int, choices=[0, 1, 2])
    parser.add_argument('--seed', type=int, choices=[17, 23, 29])
    a = parser.parse_args()
    cfg = configure(a.mode in ('witness', 'prepare', 'prepare_repair', 'preflight', 'worker'))
    out = Path(cfg['output'])
    if a.mode in ('cpu_witness', 'witness'):
        witness(cfg, a.witness_output or out / 'RUNTIME_WITNESS.json', a.mode == 'cpu_witness')
    elif a.mode in ('prepare', 'prepare_repair'):
        prepare(cfg, a.mode == 'prepare_repair')
    elif a.mode == 'preflight':
        preflight(cfg)
    elif a.mode == 'seal':
        seal(cfg)
    elif a.mode == 'worker':
        assert a.protocol in cfg['protocols'] and a.inner in cfg['inners'] and a.seed in cfg['seeds']
        worker(cfg, a.protocol, a.inner, a.seed)
    elif a.mode == 'queue':
        queue(cfg)
    else:
        check(cfg, True)
        assert not any((out / name).exists() for name in ['DISPATCH.json', 'LAUNCH.json'])
        with (out / 'queue.log').open('wb') as handle:
            child = subprocess.Popen([sys.executable, '-s', '-u', '-m', MODULE, '--mode', 'queue'], stdin=subprocess.DEVNULL,
                                     stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
        save(out / 'DISPATCH.json', dict(pid=child.pid, unix=time.time()))
        print(child.pid)


if __name__ == '__main__':
    main()
