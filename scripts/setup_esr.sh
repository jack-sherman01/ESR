#!/usr/bin/env bash
# Create the ESR environment without modifying system Python.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ESR_PYTHON:-python3.12}"
case "${1:-install}" in
  -h|--help) echo "Usage: $0 [install|smoke [NEW_OUTPUT_DIR]|ml]"; exit 0 ;;
  install)
    command -v uv >/dev/null || { echo 'Install uv first.' >&2; exit 1; }
    "$PY" -c 'import platform,sys; sys.exit(0 if sys.version_info[:2]==(3,12) and platform.system()=="Linux" and platform.machine()=="x86_64" else "Expected Linux x86_64 / Python 3.12 for this lock")'
    unset VIRTUAL_ENV PYTHONPATH PYTHONHOME
    uv venv --allow-existing --python "$PY" "$ROOT/.venv"
    uv pip install --python "$ROOT/.venv/bin/python" -c "$ROOT/configs/deployment/esr-constraints.txt" -e "$ROOT[sim]"
    ;;
  smoke)
    CDPATH= cd "$ROOT"
    out="${2:-"$ROOT/.esr-local/esr/smoke-$(date -u +%Y%m%dT%H%M%SZ)-$$"}"
    if [[ -e "$out" || -L "$out" ]]; then echo "Choose a new output directory: $out" >&2; exit 1; fi
    unset PYTHONPATH PYTHONHOME
    export MUJOCO_GL=disable OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    "$ROOT/.venv/bin/python" -B experiments/run_benchmark.py \
      --scenarios tool_misuse --gate-planner impact --generations 1 \
      --rollouts 3 --fresh-min 1 --out-dir "$out"
    "$ROOT/.venv/bin/python" -B -m esrb.detection.gate_audit "$out/tool_misuse"
    ;;
  ml)
    uv pip install --python "$ROOT/.venv/bin/python" --torch-backend cpu \
      -c "$ROOT/configs/deployment/esr-constraints.txt" 'torch==2.15.0'
    ;;
  *) echo "Unknown subcommand: $1 (see --help)" >&2; exit 2 ;;
esac
