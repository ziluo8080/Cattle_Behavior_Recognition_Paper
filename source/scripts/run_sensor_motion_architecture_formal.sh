#!/usr/bin/env bash
set -euo pipefail
cd /q3/mmcows-sensor-behavior-20261007
export PYTHONPATH=/q3/mmcows-sensor-behavior-20261007/.sensor_deps:/q3/mmcows-sensor-behavior-20261007
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=/q3/mmcows-rgbs-convnext/.venv/bin/python
if [ -e reports/sensor_motion_architecture_formal_attempt2_20261009 ]; then
  echo 'Existing official ablation attempt preserved' >&2
  exit 66
fi
set +e
REMAINING=$("$PY" -B -c 'import json,time; from pathlib import Path; c=json.loads(Path("configs/sensor_motion_architecture_formal_20261009.json").read_text()); s=json.loads(Path(c["resource_amendment"]["original_budget_start"]).read_text())["unix"]; print(max(1,int(36000-(time.time()-s))))')
timeout --signal=KILL "${REMAINING}s" "$PY" -B -m scripts.sensor_motion_architecture_formal
CODE=$?
set -e
"$PY" -B - "$CODE" <<'PY'
import json,sys,time
from pathlib import Path
p=Path('reports/sensor_motion_architecture_formal_attempt2_20261009');p.mkdir(parents=True,exist_ok=True)
r=dict(exit_code=int(sys.argv[1]),unix=time.time(),complete=int(sys.argv[1])==0,termination_metadata_only=True,budget_reset=False)
(p/'LAUNCHER_EXIT.json').write_text(json.dumps(r,indent=2))
if int(sys.argv[1]) and not (p/'FAILURE.json').exists():(p/'FAILURE.json').write_text(json.dumps(r,indent=2))
PY
exit "$CODE"
