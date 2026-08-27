#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
aa_root="${ACTIVE_ADAPTATION_ROOT:-${repo_root}/third_party/active-adaptation-dev}"
mimic_root="${repo_root}/third_party/mimic-lite"
venv_project="${MIMICLITE_VENV_PROJECT:-${repo_root}/environments/rl/mjlab}"

if [[ ! -f "${aa_root}/active_adaptation/__init__.py" ]]; then
  echo "active-adaptation is missing or incomplete: ${aa_root}" >&2
  echo "Clone branch dev/hdmi there before training." >&2
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "MimicLite mjlab training requires a visible NVIDIA CUDA GPU." >&2
  exit 3
fi

python "${repo_root}/scripts/training/build_mimiclite_71cm_motion.py"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${repo_root}/.cache/uv}"

num_envs="${MIMICLITE_NUM_ENVS:-256}"
total_frames="${MIMICLITE_TOTAL_FRAMES:-10000000}"
wandb_mode="${WANDB_MODE:-disabled}"
checkpoint_path="${MIMICLITE_CHECKPOINT_PATH:-run:elijahgalahad/mimic_lite/xua2csee}"

cd "${aa_root}"
exec uv --project "${venv_project}" run "${mimic_root}/scripts/train.py" \
  task=tracking-base \
  task/motion=g1/climb_turn_sit_71cm \
  +exp=ppo/train \
  algo/ppo/module=huge \
  backend=mjlab \
  task.terrain=box71 \
  task.num_envs="${num_envs}" \
  total_frames="${total_frames}" \
  checkpoint_path="${checkpoint_path}" \
  wandb.mode="${wandb_mode}" \
  "$@"
