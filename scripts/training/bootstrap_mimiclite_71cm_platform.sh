#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment="${repo_root}/src/longship/rl/experiments/mimiclite_g1_71cm_climb_turn_sit.yaml"
venv_project="${repo_root}/environments/rl/mjlab"
output_dir="${1:?usage: bootstrap_mimiclite_71cm_platform.sh OUTPUT_DIR}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.cache/uv}"
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-4}"

# A manually started first-time sync may already own the uv project lock.
# Wait for it instead of failing the queued platform run on lock contention.
while pgrep -f "^uv sync --project ${venv_project}$" >/dev/null 2>&1; do
  sleep 10
done

cd "${repo_root}"
uv sync --project "${venv_project}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
exec python -m longship.rl.training \
  --root "${repo_root}" \
  run "${experiment}" \
  --output "${output_dir}"
