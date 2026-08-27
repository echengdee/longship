#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LONGSHIP_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${LONGSHIP_ROOT}/src:${LONGSHIP_ROOT}/third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}"

if [[ $# -lt 1 ]]; then
  printf 'Usage: %s <deployment-profile> [options]\n' "$0" >&2
  exit 2
fi

python_ready() {
  local candidate="$1"
  [[ -x "${candidate}" ]] && "${candidate}" -c \
    'import cv2, cyclonedds, onnxruntime, zmq, unitree_sdk2py' \
    >/dev/null 2>&1
}

if [[ -n "${LONGSHIP_PYTHON:-}" ]]; then
  PYTHON_BIN="${LONGSHIP_PYTHON}"
elif python_ready "${LONGSHIP_ROOT}/.venv/bin/python"; then
  PYTHON_BIN="${LONGSHIP_ROOT}/.venv/bin/python"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  PYTHON_BIN="${CONDA_BASE}/envs/longship-rl/bin/python"
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if ! python_ready "${PYTHON_BIN}"; then
  printf 'Deployment Python is missing a required controller package.\n' >&2
  printf 'Create/update longship-rl, or set LONGSHIP_PYTHON to the correct environment.\n' >&2
  exit 2
fi

exec "${PYTHON_BIN}" -m longship.rl.deploy "$1" \
  --root "${LONGSHIP_ROOT}" \
  --python "${PYTHON_BIN}" \
  "${@:2}"
