#!/usr/bin/env bash
set -euo pipefail

BACKEND="${1:?backend is required}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LONGSHIP_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${LONGSHIP_ROOT}/src:${LONGSHIP_ROOT}/third_party/GR00T-WholeBodyControl/external_dependencies/unitree_sdk2_python${PYTHONPATH:+:${PYTHONPATH}}"

python_ready() {
  local candidate="$1"
  [[ -x "${candidate}" ]] && "${candidate}" -c \
    'import cyclonedds, mujoco, onnxruntime, zmq, unitree_sdk2py' \
    >/dev/null 2>&1
}

if [[ -n "${LONGSHIP_PYTHON:-}" ]]; then
  PYTHON_BIN="${LONGSHIP_PYTHON}"
  if ! python_ready "${PYTHON_BIN}"; then
    echo "LONGSHIP_PYTHON does not provide the required Sim2Sim packages: ${PYTHON_BIN}" >&2
    exit 2
  fi
elif [[ -n "${CONDA_PREFIX:-}" ]] && python_ready "${CONDA_PREFIX}/bin/python"; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
elif python_ready "${LONGSHIP_ROOT}/.venv/bin/python"; then
  PYTHON_BIN="${LONGSHIP_ROOT}/.venv/bin/python"
elif command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  if python_ready "${CONDA_BASE}/envs/longship-rl/bin/python"; then
    PYTHON_BIN="${CONDA_BASE}/envs/longship-rl/bin/python"
  elif python_ready "${CONDA_BASE}/envs/env_isaaclab511/bin/python"; then
    PYTHON_BIN="${CONDA_BASE}/envs/env_isaaclab511/bin/python"
  elif python_ready "$(command -v python3)"; then
    PYTHON_BIN="$(command -v python3)"
  else
    PYTHON_BIN=""
  fi
else
  PYTHON_BIN="$(command -v python3 || true)"
fi

if [[ -z "${PYTHON_BIN}" ]] || ! python_ready "${PYTHON_BIN}"; then
  echo "Python runtime is unavailable. Activate the Longship Conda environment or set LONGSHIP_PYTHON." >&2
  exit 2
fi

exec "${PYTHON_BIN}" -m longship.rl.sim2sim.runner \
  "${BACKEND}" \
  --root "${LONGSHIP_ROOT}" \
  --python "${PYTHON_BIN}"
