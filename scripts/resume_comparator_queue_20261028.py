"""Resource admission only: preserve sealed V2/V3 science and all failed receipts."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess, sys, time

CONFIG = Path('configs/sensor_comparator_queue_resume_20261028_v4.json')
V2 = 'scripts.run_comparator_convergence_20261027_v2'
V3 = 'scripts.run_comparator_joint_regularized_20261027_v3'
AUDITOR = 'scripts.audit_comparator_joint_acceptance_20261027'
RESOURCE_ERROR = 'RuntimeError: GPU occupied; no unrelated process will be stopped'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix('.tmp'); temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'); temp.replace(p)
def require(condition, message):
    if not condition: raise RuntimeError(message)
def verify(files):
    for p, digest in files.items(): require(sha(p) == digest, 'Hash mismatch: ' + str(p))
def folder(cfg, case):
    arm, protocol, inner, seed = case
    return Path(cfg['output']) / 'trajectories' / protocol / f'inner{inner}' / arm / f'seed{seed}'
def untouched(path): return not path.exists() or not any(p.is_file() for p in path.rglob('*'))
def resource_conflict(returncode, log, path):
    return returncode != 0 and RESOURCE_ERROR in Path(log).read_text(errors='replace') and untouched(path)

def runtime_code(cfg):
    return {str(p):sha(p) for p in [CONFIG,Path(__file__),Path('tests/test_comparator_queue_resume_20261028.py'),Path(cfg['ledger'])]}
def complete(path, case=None):
    receipt = read(path / 'COMPLETE.json')
    require(receipt['status'] == 'COMPLETE' and bool(receipt['files']), 'Incomplete receipt')
    if case: require([receipt[k] for k in ['model', 'protocol', 'inner', 'seed']] == list(case), 'Case identity mismatch')
    verify(receipt['files'])
    return receipt
def command(module, mode, case=None):
    cmd = [sys.executable, '-s', '-u', '-m', module, '--mode', mode]
    if case:
        for key, value in zip(['model', 'protocol', 'inner', 'seed'], case): cmd += ['--' + key, str(value)]
    return cmd
def select_pending(cfg, cases):
    require(len(cases) == len(set(map(tuple, cases))), 'Duplicate cases')
    pending = []
    for case in cases:
        p = folder(cfg, case)
        if (p / 'COMPLETE.json').exists(): complete(p, case)
        else:
            require(untouched(p), 'Existing training cannot be retried: ' + str(p)); pending.append(case)
    return pending


class Coordinator:
    def __init__(self, cfg):
        self.cfg = cfg; self.out = Path(cfg['output']); self.out.mkdir(parents=True, exist_ok=True)
        self.v2, self.v3 = read(cfg['v2_config']), read(cfg['v3_config'])
        self.started = time.monotonic(); self.deadline = self.started + cfg['queue_seconds']; self.events = []
    def state(self, status, **extra):
        save(self.out / 'QUEUE_STATE.json', dict(status=status, updated_unix=time.time(), elapsed_seconds=time.monotonic()-self.started,
             events=self.events, automatic_expansion=False, all72_completed=False, formal_results_replaced=False, **extra))
    def budget(self): require(time.monotonic() < self.deadline, '48h coordinator budget exhausted')
    def wait(self, stage):
        while True:
            self.budget(); proc = Path('/proc') / str(self.cfg['external_controller_pid']) / 'cmdline'
            try:
                command_text=proc.read_bytes().decode(errors='replace')
                stat=(proc.parent/'stat').read_text(); state=stat[stat.rfind(')')+2:].split()[0]
                busy=bool(command_text) and state!='Z'
                if busy:require(self.cfg['external_controller_command'] in command_text, 'External PID identity changed; inspect manually')
            except FileNotFoundError:busy=False
            memory = [int(x) for x in subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True).splitlines()]
            require(bool(memory), 'No GPU inventory')
            if not busy and max(memory) < self.cfg['gpu_memory_threshold_MiB']: return
            self.state('WAITING_RESOURCE', stage=stage, external_controller_alive=busy, memory_MiB=memory)
            print('WAITING_RESOURCE', stage, busy, memory, flush=True)
            time.sleep(min(self.cfg['poll_seconds'], max(0, self.deadline-time.monotonic())))
    def build(self):
        witness=read(self.out/'SYNTHETIC_WITNESS.json')
        require(witness['status']=='PASS' and witness['code']==runtime_code(self.cfg),'Runtime witness stale or absent')
        target = self.out / 'MANIFEST.json'
        if target.exists():
            manifest = read(target); verify(manifest['bindings']); return manifest
        qpath = Path(self.v2['output']) / 'QUEUE_STATE.json'; q = read(qpath)
        require(sha(qpath) == self.cfg['source_queue_sha256'] and q['status'] == 'FAILED_RUNTIME_NO_RETRY', 'Unexpected source failure state')
        failed = [j for j in q['jobs'] if j['status'] == 'FAILED_RUNTIME']
        require(len(failed) == 1 and sha(failed[0]['log']) == self.cfg['source_error_log_sha256'], 'Unexpected source failure log')
        bindings = {str(p): sha(p) for p in [CONFIG, Path(__file__), qpath, Path(failed[0]['log']), Path(self.cfg['v2_config']), Path(self.cfg['v3_config'])]}
        bindings.update(runtime_code(self.cfg));bindings[str(self.out/'SYNTHETIC_WITNESS.json')]=sha(self.out/'SYNTHETIC_WITNESS.json')
        sealpath = Path(self.v2['output']) / 'SEAL.json'; seal = read(sealpath)
        require(seal['config'] == self.v2, 'V2 scientific config changed'); verify(seal['code']); verify(seal['files'])
        bindings.update(seal['code']); bindings.update(seal['files']); bindings[str(sealpath)] = sha(sealpath)
        for path in sorted((Path(self.v2['output']) / 'trajectories').glob('*/*/*/seed*/COMPLETE.json')):
            receipt = complete(path.parent); bindings[str(path)] = sha(path); bindings.update(receipt['files'])
        cpu = Path(self.v3['output']) / 'CPU_WITNESS_FINAL.json'; witness = read(cpu)
        require(witness['status'] == 'PASS', 'Missing V3 CPU witness'); verify(witness['code']); bindings.update(witness['code']); bindings[str(cpu)] = sha(cpu)
        pending = select_pending(self.v2, self.cfg['remaining_cases'])
        require(len(pending) == 8, 'Initial manifest must contain exactly eight missing cases')
        manifest = dict(bindings=bindings, cases=pending, commands=[command(V2, 'worker', c) for c in pending], v3_code=witness['code'])
        save(target, manifest); return manifest
    def invoke(self, stage, cmd, timeout, manifest, path=None, retry=False):
        for admission in range(self.cfg['maximum_resource_readmissions'] + 1):
            verify(manifest['bindings'])
            if stage=='joint_audit':self.budget()
            else:self.wait(stage)
            if path is not None: require(untouched(path), 'Existing training cannot be launched: ' + str(path))
            log = self.out / 'logs' / f'{stage}_admission{admission}_{time.time_ns()}.log'
            log.parent.mkdir(exist_ok=True); self.state('RUNNING', stage=stage, command=cmd, log=str(log))
            with log.open('xb') as stream:
                result = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, timeout=min(timeout, self.deadline-time.monotonic()))
            self.events.append(dict(stage=stage, returncode=result.returncode, log=str(log), log_sha256=sha(log), resource_admission=admission))
            if not result.returncode: return
            if retry and resource_conflict(result.returncode, log, path) and admission < self.cfg['maximum_resource_readmissions']:
                self.state('WAITING_RESOURCE', stage=stage, reason='GPU occupied before STARTED; science attempt not begun'); continue
            raise RuntimeError('Child failed; no science retry: ' + str(log))
    def audit(self, manifest):
        self.invoke('joint_audit', command(AUDITOR, 'once'), 1200, manifest)
    def run(self):
        manifest = self.build()
        for index, case in enumerate(manifest['cases']):
            p = folder(self.v2, case)
            if not (p / 'COMPLETE.json').exists():
                self.invoke('v2_' + '_'.join(map(str, case)), manifest['commands'][index], self.v2['budget']['worker_seconds']+60, manifest, p, True)
            receipt = complete(p, case)
            save(self.out / ('V2_COMPLETE_' + str(index) + '.json'), dict(case=case, source_sha256=sha(p/'COMPLETE.json'), files=receipt['files']))
            self.audit(manifest)
        for mode, filename in [('cuda_witness', 'CUDA_WITNESS.json'), ('preflight', 'PREFLIGHT.json'), ('prepare', 'SEAL.json')]:
            p = Path(self.v3['output']) / filename
            if not p.exists(): self.invoke('v3_' + mode, command(V3, mode), 1200, manifest)
            receipt = read(p); require(receipt['code'] == manifest['v3_code'], 'V3 prerequisite code changed'); verify(receipt['code'])
            if mode == 'prepare':
                require(receipt['config'] == self.v3, 'V3 seal config changed'); verify(receipt['files'])
            else: require(receipt['status'] == 'PASS', 'V3 prerequisite failed')
        pilots = []
        for protocol in ['TS', 'OS']:
            case = ['TSLANet_Sup_Fusion', protocol, 0, 17]; p = folder(self.v3, case)
            if not (p/'COMPLETE.json').exists(): self.invoke('v3_pilot_' + protocol, command(V3, 'worker', case), self.v3['budget']['worker_seconds']+60, manifest, p, True)
            receipt = complete(p, case); pilots.append(dict(case=case, joint_pass=receipt['empirical_convergence'], source_sha256=sha(p/'COMPLETE.json')))
            self.audit(manifest)
        verify(manifest['bindings'])
        self.state('RESUME_SCOPE_COMPLETE', v2_resumed_cases=8, v3_pilots=pilots, remaining_original_TSLA_unexpanded=16)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--mode', required=True, choices=['build', 'run', 'selftest']); args = parser.parse_args()
    if args.mode == 'selftest':
        result = subprocess.run([sys.executable, '-m', 'unittest','discover','-s','tests','-p','test_comparator_queue_resume_20261028.py','-v'])
        if result.returncode: raise SystemExit(result.returncode)
        cfg=read(CONFIG)
        save(Path(cfg['output'])/'SYNTHETIC_WITNESS.json',dict(status='PASS',real_data_rows_read=0,GPU_used=False,code=runtime_code(cfg)))
        print('SYNTHETIC_ONLY_PASS real_data_rows_read=0 GPU_used=False'); return
    cfg = read(CONFIG); require(Path.cwd() == Path(cfg['root']), 'Real workloads remain cloud-only')
    import fcntl
    coordinator = Coordinator(cfg)
    with (coordinator.out/'COORDINATOR.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            if args.mode == 'build': coordinator.build()
            else: coordinator.run()
        except Exception as exc:
            coordinator.state('FAILED_NO_SCIENCE_RETRY', error=repr(exc)); raise


if __name__ == '__main__': main()
