#!/usr/bin/env bash
set -uo pipefail
cd /q3/mmcows-sensor-behavior-20261007 || exit 2
task_root=reports/sensor_classic_with_original_rf_20261014_attempt2
mkdir -p "$task_root"
export PYTHONPATH=.classic_deps_20261012:.sensor_deps:.
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMBA_NUM_THREADS=4
export PYTHONDONTWRITEBYTECODE=1
flock -n "$task_root/.RUNNING.lock" /q3/mmcows-rgbs-convnext/.venv/bin/python -u -m scripts.extend_sensor_classic_original_rf_20261014
task_exit_code=$?
TASK_EXIT_CODE="$task_exit_code" /q3/mmcows-rgbs-convnext/.venv/bin/python - <<'PY'
import json, os, time
from pathlib import Path
root = Path('reports/sensor_classic_with_original_rf_20261014_attempt2')
receipt = dict(exit_code=int(os.environ['TASK_EXIT_CODE']),
    ended_UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    stage='original_RF_replay_and_six_model_rescore', independent_audit='pending')
(root / f'LAUNCHER_EXIT_{time.time_ns()}.json').write_text(json.dumps(receipt, indent=2))
PY
exit "$task_exit_code"
