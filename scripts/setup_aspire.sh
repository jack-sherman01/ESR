#!/usr/bin/env bash
# Optional ASPIRE simulation setup. Never installs into ESR's environment.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
KIT="$ROOT/integrations/aspire"
STATE="$ROOT/.esr-local/aspire"
ASPIRE="$(realpath -m -- "${ESR_ASPIRE_DIR:-"$STATE/checkout"}")"
RECEIPT="$STATE/sam3-receipt.json"
PY="${ESR_PYTHON:-python3.12}"
PROFILE="${CAP_HARNESS_PROFILE:-rtx4090}"

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_aspire.sh [install|source|assets|serve|smoke|check|path] [NEW_OUTPUT_DIR]
  install  Create the pinned checkout and isolated simulator/provider environments.
  source   Fetch and verify only the pinned ASPIRE source.
  assets   Download authorized SAM3 assets and pin the Panda description.
  serve    Run SAM3, Contact-GraspNet and PyRoki in the foreground.
  smoke    Run LIBERO-Pro reset/steps and Robosuite reset without providers or an Agent.
  check    Check all providers and run SAM3 on a simulated camera image.
  path     Print the ASPIRE checkout path.
EOF
}

install_source() {
  local pin patch
  patch="$KIT/aspire-local.patch"
  pin="$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["upstream_commit"])' "$KIT/aspire-source.json")"
  if [[ ! -d "$ASPIRE/.git" ]]; then
    if [[ -e "$ASPIRE" ]]; then
      echo "Existing checkout path is not an installer-owned Git directory: $ASPIRE" >&2; return 1
    fi
    mkdir -p "$ASPIRE"
    git -C "$ASPIRE" init -q
    git -C "$ASPIRE" remote add origin https://github.com/NVlabs/ASPIRE.git
  fi
  if [[ "$(git -C "$ASPIRE" rev-parse --verify HEAD 2>/dev/null || true)" != "$pin" ]]; then
    if git -C "$ASPIRE" rev-parse --verify HEAD >/dev/null 2>&1; then
      echo "Existing ASPIRE checkout has a different HEAD: $ASPIRE" >&2; return 1
    fi
    git -C "$ASPIRE" fetch --depth=1 origin "$pin"
    git -C "$ASPIRE" checkout --detach "$pin"
  fi
  if ! git -C "$ASPIRE" apply --reverse --check "$patch" 2>/dev/null; then
    git -C "$ASPIRE" apply --check "$patch"
    git -C "$ASPIRE" apply "$patch"
  fi
  "$PY" -I "$KIT/verify_aspire.py" "$ASPIRE"
}

output_dir() {
  local out
  out="$(realpath -m -- "${2:-"$STATE/$1-$(date -u +%Y%m%dT%H%M%SZ)-$$"}")"
  if [[ -e "$out" || -L "$out" ]]; then
    echo "Choose a new output directory: $out" >&2; return 1
  fi
  printf '%s\n' "$out"
}

unset VIRTUAL_ENV PYTHONPATH PYTHONHOME
case "${1:-install}" in
  -h|--help) usage ;;
  path) printf '%s\n' "$ASPIRE" ;;
  source) install_source ;;
  install)
    command -v uv >/dev/null || { echo 'Install uv first.' >&2; exit 1; }
    "$PY" -c 'import platform,sys; sys.exit(0 if sys.version_info[:2]==(3,12) and platform.system()=="Linux" and platform.machine()=="x86_64" else "Expected Linux x86_64 / Python 3.12 for this lock")'
    install_source
    export CAP_HARNESS_PROFILE="$PROFILE" CAP_HARNESS_PYTHON="$(command -v "$PY")"
    export CAP_HARNESS_ROBOSUITE_CACHE="$ASPIRE/.uv-cache"
    export LIBERO_CONFIG_PATH="$ASPIRE/.libero"
    cd "$ASPIRE"
    for name in libero robosuite sam3 graspnet pyroki; do
      bash "scripts/bootstrap_${name}.sh"
    done
    ;;
  assets)
    if [[ ! -x "$ASPIRE/.venv-sam3/bin/python" ]]; then echo 'Install ASPIRE first.' >&2; exit 1; fi
    panda="$HOME/.cache/robot_descriptions/example-robot-data"
    pin=d0d9098d752014aec3725b07766962acf06c5418
    if [[ ! -e "$panda" ]]; then
      mkdir -p "$(dirname -- "$panda")"
      git clone --depth=1 --branch v4.0.9 https://github.com/Gepetto/example-robot-data.git "$panda"
    fi
    [[ "$(git -C "$panda" rev-parse HEAD)" == "$pin" ]] || {
      echo "Panda cache has a different revision: $panda" >&2; exit 1;
    }
    test -f "$panda/robots/panda_description/urdf/panda.urdf"
    cd "$ASPIRE"
    sha256sum -c "$KIT/graspnet.sha256"
    "$ASPIRE/.venv-sam3/bin/python" -I "$KIT/prepare_sam3.py" "$RECEIPT"
    ;;
  serve)
    out="$(output_dir services "${2:-}")"
    exec "$PY" -I "$KIT/aspire_service_supervisor.py" --checkout "$ASPIRE" --receipt "$RECEIPT" \
      --sam3-precision "${ESR_ASPIRE_SAM3_PRECISION:-float32-offload}" --output-dir "$out"
    ;;
  smoke|check)
    out="$(output_dir "$1" "${2:-}")"
    mkdir -p -- "$out"
    export LIBERO_CONFIG_PATH="$ASPIRE/.libero" TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl EGL_PLATFORM=device
    export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
    cd "$ASPIRE"
    if [[ "$1" == smoke ]]; then
      "$ASPIRE/.venv-libero/bin/python" -B "$KIT/aspire_smoke.py" --checkout "$ASPIRE" --out "$out/libero"
      "$ASPIRE/.venv-robosuite/bin/cap-harness" doctor --skip-services \
        --environment-root "$ASPIRE/.venv-robosuite" --expected-environment-name .venv-robosuite \
        --runtime robosuite --reset --reset-embodiment robosuite --reset-suite cube_lifting \
        --output "$out/robosuite.json"
    else
      for port in 8114 8115 8116; do
        curl -fsS --noproxy '*' --max-time 5 "http://127.0.0.1:$port/healthz" > "$out/health-$port.json"
        "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["ok"] is True else 1)' "$out/health-$port.json"
      done
      "$ASPIRE/.venv-libero/bin/python" -B "$KIT/aspire_perception_smoke.py" --out "$out/sam3"
    fi
    ;;
  *) usage >&2; exit 2 ;;
esac
