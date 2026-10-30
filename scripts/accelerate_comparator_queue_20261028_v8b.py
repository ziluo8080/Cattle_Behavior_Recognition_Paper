"""Runtime-only bounded parallel execution of the immutable V7 DS workers.

No scientific checkpoint is resumed or recomputed here.  COMPLETE receipts are
verified, in-flight workers are adopted, and only never-started cases dispatch.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

CONFIG = Path('configs/sensor_comparator_parallel_runtime_20261028_v8b.json')
MODULE = 'scripts.accelerate_comparator_queue_20261028_v8b'
SCIENCE_MODULE = 'scripts.run_comparator_repair_20261028_v7'
OUTER_MODULE = 'scripts.continue_comparator_repair_20261028_v7'
TEST = 'tests/test_comparator_parallel_runtime_20261028_v8b.py'
OUTER = Path('reports/sensor_five_comparator_repair_20261028_v7')
KEYS = ('model', 'protocol', 'inner', 'seed')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def science_module():
    from scripts import run_comparator_repair_20261028_v7
    return run_comparator_repair_20261028_v7


def setup(runtime_config=None):
    runtime = read(runtime_config or CONFIG)
    assert runtime['science_config'] == 'configs/sensor_comparator_repair_20261028_v7.json'
    assert sha(runtime['science_config']) == runtime['science_config_sha256']
    assert runtime['GPU_processes'] in (2, 3)
    assert runtime['Torch_memory_fraction_per_process'] == .20
    assert runtime['minimum_free_GPU_MiB'] >= 6144
    original = science_module()
    science = original.setup(False)
    original.check(science)
    return science, runtime, Path(runtime['runtime_output'])


def code(runtime):
    paths = [CONFIG, Path(__file__), TEST, runtime['ledger'],
             'scripts/continue_comparator_repair_20261028_v7.py',
             'scripts/report_comparator_repair_20261028_v7.py']
    paths += runtime.get('additional_code_paths', [])
    result = {Path(p).as_posix(): sha(p) for p in paths}
    original = science_module()
    result.update(original.code(read(original.CONFIG)))
    return result


def case_id(job):
    return '/'.join(str(job[k]) for k in KEYS)


def all_jobs(science):
    return [dict(model=m, protocol=p, inner=k, seed=s, status='pending')
            for m in science['models'] for p in science['protocols']
            for k in science['inners'] for s in science['seeds']]


def folder(science, job):
    return Path(science['output']) / 'trajectories' / job['protocol'] / ('inner' + str(job['inner'])) / job['model'] / ('seed' + str(job['seed']))


def completed(science, job):
    path = folder(science, job) / 'COMPLETE.json'
    receipt = read(path)
    assert all(receipt[k] == job[k] for k in KEYS), str(path)
    assert receipt['status'] == 'COMPLETE_DEVELOPMENT'
    assert receipt['selected_checkpoint_replay_exact']
    assert receipt['official_Test_rows_read'] == 0
    required = {str(folder(science, job) / n) for n in
                ['history.json', 'selected_best_DEV_CE.pt', 'selected_DEV_EMA.npy', 'latest.pt']}
    assert set(receipt['files']) == required, str(path)
    for name, digest in receipt['files'].items():
        assert sha(name) == digest, name
    return receipt


def eligible(job, receipts, protocols):
    if job['status'] != 'pending':
        return False
    if (job['inner'], job['seed']) == (0, 17):
        return True
    from scripts.comparator_repair_policy_20261028 import allow_confirmation
    pilots = [receipts.get(case_id(dict(model=job['model'], protocol=p, inner=0, seed=17))) for p in protocols]
    return all(p is not None for p in pilots) and allow_confirmation(pilots)


def process(pid):
    """Read stable Linux identity; zombies are dead, PID reuse is a different identity."""
    try:
        base = Path('/proc') / str(pid)
        first = (base / 'stat').read_text()
        fields = first[first.rfind(')') + 2:].split()
        tokens = (base / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
        second = (base / 'stat').read_text()
        check = second[second.rfind(')') + 2:].split()
        if fields[19] != check[19] or check[0] == 'Z':
            return None
        boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
        return dict(pid=int(pid), start_ticks=int(check[19]), ppid=int(check[1]), state=check[0],
                    pgid=int(check[2]), sid=int(check[3]),
                    started_unix=boot + int(check[19]) / os.sysconf('SC_CLK_TCK'), tokens=tokens)
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_identity(expected, current):
    return current is not None and all(expected[k] == current[k] for k in ('pid', 'start_ticks', 'tokens'))


def module_mode(identity, module, mode=None):
    if not identity:
        return False
    tokens = identity['tokens']
    if '-m' not in tokens or tokens[tokens.index('-m') + 1] != module:
        return False
    if mode is None:
        return '--mode' not in tokens
    return '--mode' in tokens and tokens[tokens.index('--mode') + 1] == mode


def parent_identity(launch, module, mode=None):
    identity = process(launch['pid'])
    assert module_mode(identity, module, mode), ('Wrong/dead parent', launch, identity)
    # LAUNCH is written after imports. A birth later than LAUNCH is always reuse.
    assert identity['started_unix'] <= launch['unix'] + 1
    assert launch['unix'] - identity['started_unix'] < 300
    return identity


def worker_job(identity):
    assert module_mode(identity, SCIENCE_MODULE, 'worker')
    tokens = identity['tokens']
    job = {k: tokens[tokens.index('--' + k) + 1] for k in KEYS}
    job['inner'], job['seed'] = int(job['inner']), int(job['seed'])
    return job


def expected_gpu_query(identity):
    tokens=identity['tokens']
    return (bool(tokens) and Path(tokens[0]).name=='nvidia-smi' and
            tokens[1:]==['--query-gpu=memory.used','--format=csv,noheader,nounits'])


def rollback_parents(stopped, committed, reader=process, sender=os.kill):
    """Only verified parents are resumed before the termination commit."""
    if not committed:
        for identity in reversed(stopped):
            if same_identity(identity, reader(identity['pid'])):
                sender(identity['pid'], signal.SIGCONT)


def retirement_safe(active, reader=process):
    """A screen session leader must outlive every adopted scientific child."""
    return all(not same_identity(a['identity'], reader(a['identity']['pid'])) for a in active)


def retire_parents(stopped, active, reader=process, sender=os.kill, sleeper=time.sleep):
    assert retirement_safe(active, reader), 'Do not retire a screen parent while its scientific child is alive'
    for parent in stopped:
        current = reader(parent['pid'])
        if current is not None:
            assert same_identity(parent, current), 'Old parent PID reused; never signal the replacement'
            try:
                sender(parent['pid'], signal.SIGTERM)
                if current.get('state') in ('T','t'):
                    sender(parent['pid'], signal.SIGCONT)
            except ProcessLookupError:
                pass
    for _ in range(100):
        if all(not same_identity(p, reader(p['pid'])) for p in stopped):
            return
        sleeper(.05)
    raise RuntimeError('Verified old parent still alive after retirement')


def snapshot_jobs(science, active, receipts):
    rows = all_jobs(science)
    lookup = {case_id(j): j for j in rows}
    assert len(lookup) == len(rows)
    assert len(active) <= 1
    for a in active:
        key = case_id(a['job'])
        assert key in lookup
        lookup[key].update(status='running', identity=a['identity'], adopted=True, slot=0,
                           started_unix=a['started_unix'], receipt_at_handoff=key in receipts)
    for key in receipts:
        assert key in lookup
        if lookup[key]['status'] != 'running':
            lookup[key]['status'] = 'completed'
    return rows


def lock_file(path):
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def slot_gpu(runtime, slot):
    import torch
    assert slot in range(runtime['GPU_processes'])
    handle = lock_file(Path(runtime['runtime_output']) / ('GPU_SLOT_%d.lock' % slot))
    try:
        free = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free',
                   '--format=csv,noheader,nounits'], text=True).splitlines()[0])
        assert free >= runtime['minimum_free_GPU_MiB'], 'Resource admission failed before scientific STARTED'
        torch.cuda.set_per_process_memory_fraction(runtime['Torch_memory_fraction_per_process'])
        return handle
    except BaseException:
        handle.close()
        raise


def cpu_witness(science, runtime, out):
    import torch
    assert not torch.cuda.is_available(), 'Hide CUDA for this documented CPU witness'
    assert not (out / 'CPU_WITNESS.json').exists()
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests',
                    '-p', Path(TEST).name], check=True)
    save(out / 'CPU_WITNESS.json', dict(status='PASS', code=code(runtime), real_data_rows_read=0,
         slots_mutually_exclusive=True, handoff_identity_rollback_races_tested=True, scientific_changes=[]))
    print('COMPARATOR_PARALLEL_RUNTIME_CPU_PASS', flush=True)


def profile_path(runtime):
    return Path(runtime.get('profile_result', 'profile_output/comparator_gpu_20261028_v8/RESULTS.json'))


def seal(science, runtime, out):
    assert not (out / 'RUNTIME_SEAL.json').exists()
    c = code(runtime)
    witness = read(out / 'CPU_WITNESS.json')
    assert witness['status'] == 'PASS' and witness['code'] == c
    p = read(profile_path(runtime))
    assert p['status'] == 'PASS' and p['parallel_vs_serial_bitwise_exact'] is True
    assert p['speedup'] >= 1.1
    assert p['max_process_peak_mib'] < 24576 * runtime['Torch_memory_fraction_per_process']
    assert p['source_hashes']
    for path, digest in p['source_hashes'].items():
        assert sha(path) == digest, path
    assert p['source_hashes'][runtime['science_config']] == runtime['science_config_sha256']
    save(out / 'RUNTIME_SEAL.json', dict(status='PASS', runtime=runtime, code=c,
         science_seal_sha256=sha(Path(science['output']) / 'SEAL.json'),
         cpu_witness_sha256=sha(out / 'CPU_WITNESS.json'), profile_sha256=sha(profile_path(runtime)),
         scientific_changes=[], unix=time.time()))


def check(science, runtime, out):
    s = read(out / 'RUNTIME_SEAL.json')
    assert s['status'] == 'PASS' and s['runtime'] == runtime and s['code'] == code(runtime)
    assert s['science_seal_sha256'] == sha(Path(science['output']) / 'SEAL.json')
    assert s['cpu_witness_sha256'] == sha(out / 'CPU_WITNESS.json')
    assert s['profile_sha256'] == sha(profile_path(runtime))


def handoff(science, runtime, out):
    root = Path(science['output'])
    assert not (out / 'HANDOFF_PREPARED.json').exists()
    launch_outer, launch_inner = read(OUTER / 'LAUNCH.json'), read(root / 'LAUNCH.json')
    assert launch_outer['coordinator_sha256'] == sha('scripts/continue_comparator_repair_20261028_v7.py')
    outer = parent_identity(launch_outer, OUTER_MODULE)
    inner = parent_identity(launch_inner, SCIENCE_MODULE, 'queue')
    assert inner['ppid'] == outer['pid'], 'Coordinator/queue ancestry mismatch'
    assert os.getsid(0) != outer['sid'] and os.getpgrp() != outer['pgid'], 'New controller must be outside old terminal session'
    stopped, committed = [], False
    try:
        # screen auto-continues its direct child; freeze only the inner dispatcher.
        for parent in (inner,):
            assert same_identity(parent, process(parent['pid']))
            os.kill(parent['pid'], signal.SIGSTOP)
            stopped.append(parent)
        for parent in stopped:
            for _ in range(50):
                current = process(parent['pid'])
                assert same_identity(parent, current)
                if current['state'] in ('T', 't'):
                    break
                time.sleep(.02)
            else:
                raise RuntimeError('Parent did not stop')
        for label, path in [('OUTER_LAUNCH', OUTER/'LAUNCH.json'), ('OUTER_QUEUE', OUTER/'QUEUE_STATE.json'),
                            ('INNER_LAUNCH', root/'LAUNCH.json'), ('INNER_QUEUE', root/'QUEUE_STATE.json')]:
            target = out / (label + '.json')
            assert not target.exists()
            target.write_bytes(path.read_bytes())
        children = Path('/proc/%d/task/%d/children' % (inner['pid'], inner['pid'])).read_text().split()
        active = []
        for pid in children:
            identity = process(int(pid))
            if identity is None:
                continue
            birth=identity['start_ticks']
            for _ in range(100):
                if module_mode(identity,SCIENCE_MODULE,'worker') or expected_gpu_query(identity):
                    break
                time.sleep(.02)
                identity=process(int(pid))
                if identity is None:
                    break
                assert identity['start_ticks']==birth, 'Child PID reused during fork/exec observation'
            if identity is None:
                continue
            assert identity['ppid'] == inner['pid']
            if expected_gpu_query(identity):
                for _ in range(100):
                    if not same_identity(identity, process(identity['pid'])):
                        break
                    time.sleep(.05)
                else:
                    raise RuntimeError('Transient GPU query did not finish; rollback parent freeze')
                continue
            job = worker_job(identity)
            started_path = folder(science, job) / 'STARTED.json'
            started = read(started_path) if started_path.exists() else None
            if started:
                assert started['pid'] == identity['pid']
                assert identity['started_unix'] <= started['unix'] + 1
            active.append(dict(identity=identity, job=job,
                               started_unix=started['unix'] if started else identity['started_unix']))
        receipts = {}
        for job in all_jobs(science):
            d = folder(science, job)
            if (d/'COMPLETE.json').exists():
                receipts[case_id(job)] = completed(science, job)
            elif d.exists():
                assert any(case_id(a['job']) == case_id(job) for a in active), 'Unfinished orphan: ' + str(d)
        rows = snapshot_jobs(science, active, receipts)
        save(out/'HANDOFF_PREPARED.json', dict(parents=[outer,inner], suspended_dispatchers=stopped, active=active, rows=rows,
             original_queue_started_unix=launch_inner['unix'], unix=time.time(), scientific_changes=[]))
        # After committing new ownership, never resume the old dispatcher.
        for parent in (outer,inner):
            assert same_identity(parent, process(parent['pid']))
        save(out/'HANDOFF_OWNERSHIP_COMMITTED.json',dict(controller_pid=os.getpid(),parents=[outer,inner],parents_suspended=stopped,
             adopted_scientific_children=active,original_queue_started_unix=launch_inner['unix'],unix=time.time(),
             recovery_rule='After this marker do not resume old dispatchers; preserve pending/running cases for explicit recovery',
             scientific_worker_signals_sent=0))
        committed = True
        parents_retired = retirement_safe(active)
        if parents_retired:
            retire_parents([outer,inner], active)
        save(out/'HANDOFF_COMPLETE.json', dict(status='PASS', parents=[outer,inner], suspended_dispatchers=stopped, active=active,
             controller_pid=os.getpid(), original_queue_started_unix=launch_inner['unix'],
             worker_signals_sent=0, parents_retired=parents_retired,
             deferred_retirement_reason='Preserve old screen session until adopted child naturally exits', unix=time.time()))
        save(root/'RUNTIME_OVERRIDE_STATE.json', dict(status='ACTIVE_RUNTIME_OVERRIDE', runtime=MODULE,
             runtime_state=str(out/'QUEUE_STATE.json'), runtime_seal_sha256=sha(out/'RUNTIME_SEAL.json'),
             original_LAUNCH_preserved=True, original_queue_started_unix=launch_inner['unix'],
             scientific_changes=[], unix=time.time()))
        return rows, receipts, launch_inner['unix']
    except BaseException:
        save(out/'HANDOFF_FAILURE.json', dict(committed=committed, stopped=stopped,
             traceback=traceback.format_exc(), unix=time.time()))
        raise
    finally:
        rollback_parents(stopped, committed)


def controller(science, runtime, out):
    check(science, runtime, out)
    guard = lock_file(out/'CONTROLLER.lock')
    rows, receipts, queue_start = handoff(science, runtime, out)
    handoff_record = read(out/'HANDOFF_COMPLETE.json')
    parents_retired = handoff_record['parents_retired']
    owned, streams = {}, {}
    reservation = lock_file(out/'GPU_SLOT_0.lock') if any(j.get('adopted') for j in rows) else None
    shared = None
    failed, errors = False, []
    save(out/'CONTROLLER_STARTED.json', dict(pid=os.getpid(), unix=time.time(), original_queue_started_unix=queue_start))
    try:
        while True:
            for job in rows:
                if job['status'] != 'running':
                    continue
                key = case_id(job)
                handle = owned.get(key)
                if handle is not None:
                    running = handle.poll() is None
                    if job.get('identity') is None and running:
                        identity = process(handle.pid)
                        if identity is not None:
                            job['identity'] = identity
                else:
                    running = same_identity(job['identity'], process(job['identity']['pid']))
                started_path = folder(science, job)/'STARTED.json'
                if started_path.exists() and not job.get('start_receipt_observed'):
                    receipt = read(started_path)
                    assert receipt['pid'] == (handle.pid if handle is not None else job['identity']['pid'])
                    job['started_unix'] = receipt['unix']
                    job['start_receipt_observed'] = True
                if running:
                    if time.time()-job['started_unix'] > science['budget']['worker_seconds']+60 and not job.get('timed_out'):
                        job['timed_out'] = True
                        failed = True
                        errors.append(dict(job=key, error='WORKER_TIMEOUT_DRAIN_WITHOUT_SIGNALS'))
                    continue
                try:
                    if handle is not None:
                        assert handle.returncode == 0, 'Worker returned ' + str(handle.returncode)
                    receipts[key] = completed(science, job)
                    job['status'] = 'completed'
                except BaseException:
                    job['status'] = 'failed'
                    failed = True
                    errors.append(dict(job=key, traceback=traceback.format_exc()))
                if key in streams:
                    streams.pop(key).close()
                if job.get('adopted') and reservation is not None:
                    reservation.close()
                    reservation = None
            running_jobs = [j for j in rows if j['status'] == 'running']
            if not parents_retired and reservation is None:
                retire_parents(handoff_record['parents'], handoff_record['active'])
                parents_retired = True
                save(out/'PARENTS_RETIRED.json', dict(status='PASS', unix=time.time(),
                     after_adopted_scientific_child_exit=True, scientific_worker_signals_sent=0))
            if time.time()-queue_start >= science['budget']['queue_seconds'] and not failed:
                failed = True
                errors.append(dict(error='ORIGINAL_QUEUE_BUDGET_EXHAUSTED_NO_RESET'))
            # Preserve the project's existing global exclusion once adopted worker releases it.
            if reservation is None and shared is None and not failed:
                try:
                    shared = lock_file('reports/sensor_motion_sync_validation_20261020_v1/SINGLE_GPU.lock')
                except BlockingIOError:
                    failed = True
                    errors.append(dict(error='GLOBAL_GPU_LOCK_OWNED_BY_ANOTHER_TASK'))
            if not failed:
                free_slots = sorted(set(range(runtime['GPU_processes'])) - {j['slot'] for j in running_jobs})
                for job in rows:
                    if not free_slots:
                        break
                    if not eligible(job, receipts, science['protocols']):
                        continue
                    assert not folder(science, job).exists(), 'Never restart an existing science folder'
                    slot = free_slots.pop(0)
                    key = case_id(job)
                    logfile = out / ('worker_' + key.replace('/', '_') + '.log')
                    stream = logfile.open('xb')
                    args = [sys.executable, '-s', '-u', '-m', MODULE, '--mode', 'worker', '--slot', str(slot)]
                    for k in KEYS:
                        args += ['--'+k, str(job[k])]
                    try:
                        process_handle = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT,
                                                          stdin=subprocess.DEVNULL, start_new_session=True)
                    except BaseException:
                        stream.close()
                        raise
                    owned[key], streams[key] = process_handle, stream
                    job.update(status='running', slot=slot, pid=process_handle.pid, identity=process(process_handle.pid),
                               started_unix=time.time(), adopted=False, log=str(logfile))
                    # Publish each one-time dispatch before starting another subprocess.
                    save(out/'DISPATCH_STATE.json', dict(rows=rows, unix=time.time()))
            running_jobs = [j for j in rows if j['status'] == 'running']
            status = 'FAILED_DRAINING_NO_NEW_DISPATCH' if failed and running_jobs else 'FAILED_NO_BLIND_RETRY' if failed else 'RUNNING'
            state = dict(status=status, jobs=rows, completed=sum(j['status']=='completed' for j in rows),
                         active=len(running_jobs), errors=errors, original_queue_started_unix=queue_start,
                         scientific_changes=[], formal_fivefold_complete=False, unix=time.time())
            save(out/'QUEUE_STATE.json', state)
            if not running_jobs:
                if failed:
                    break
                if not any(eligible(j, receipts, science['protocols']) for j in rows):
                    break
            time.sleep(runtime.get('poll_seconds', 5))
        if failed:
            save(Path(science['output'])/'RUNTIME_OVERRIDE_STATE.json', state)
            raise RuntimeError('Runtime failed; running workers drained and scientific evidence retained')
        check(science, runtime, out)
        for job in rows:
            if job['status']=='completed':
                completed(science, job)
        subprocess.run([sys.executable, '-m', 'scripts.report_comparator_repair_20261028_v7'], check=True, timeout=300)
        state.update(status='COMPLETE_REVIEW_REQUIRED_RUNTIME_OVERRIDE', runtime_override=MODULE,
                     completed=sum(j['status']=='completed' for j in rows),
                     blocked=sum(j['status']=='pending' for j in rows), legacy_gates_changed=False,
                     independent_generalization=False)
        save(out/'COMPLETE.json', state)
        save(out/'QUEUE_STATE.json', state)
        save(Path(science['output'])/'RUNTIME_OVERRIDE_STATE.json', state)
        save(Path(science['output'])/'COMPLETE.json', state)
        save(Path(science['output'])/'QUEUE_STATE.json', state)
        prior = read(out/'OUTER_QUEUE.json')
        save(OUTER/'QUEUE_STATE.json', dict(status='REPAIR_BATCH_COMPLETE_REVIEW_REQUIRED_RUNTIME_OVERRIDE',
             prior_stages=prior.get('stages', []), runtime_override=MODULE, runtime_completion=str(out/'COMPLETE.json'),
             formal_fivefold_complete=False, unix=time.time()))
    except BaseException:
        save(out/'CONTROLLER_FAILURE.json', dict(traceback=traceback.format_exc(), jobs=rows, unix=time.time(),
             surviving_workers_not_signalled=True))
        raise
    finally:
        for stream in streams.values():
            stream.close()
        if reservation is not None:
            reservation.close()
        if shared is not None:
            shared.close()
        guard.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['cpu_witness', 'seal', 'worker', 'queue'], required=True)
    for key in KEYS:
        parser.add_argument('--'+key, type=int if key in ('inner', 'seed') else str)
    parser.add_argument('--slot', type=int)
    args = parser.parse_args()
    science, runtime, out = setup()
    if args.mode == 'cpu_witness':
        cpu_witness(science, runtime, out)
    elif args.mode == 'seal':
        seal(science, runtime, out)
    elif args.mode == 'queue':
        controller(science, runtime, out)
    else:
        check(science, runtime, out)
        job = {k: getattr(args, k) for k in KEYS}
        assert case_id(job) in {case_id(j) for j in all_jobs(science)}
        original = science_module()
        original.setup(True)
        original.core.acquire_gpu = lambda: slot_gpu(runtime, args.slot)
        original.worker(science, **job)


if __name__ == '__main__':
    main()
