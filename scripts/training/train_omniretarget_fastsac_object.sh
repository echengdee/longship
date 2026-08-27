#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LONGSHIP_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
HOLOSOMA_ROOT="$LONGSHIP_ROOT/third_party/holosoma"

HOLOSOMA_TRAIN_PYTHON="${LONGSHIP_HOLOSOMA_TRAIN_PYTHON:-/home/qcraft/miniconda3/envs/env_isaaclab511/bin/python}"
OMNIRETARGET_NUM_ENVS="${LONGSHIP_OMNIRETARGET_NUM_ENVS:-512}"
OMNIRETARGET_TRAIN_STEPS="${LONGSHIP_OMNIRETARGET_TRAIN_STEPS:-400000}"
OMNIRETARGET_SAVE_INTERVAL="${LONGSHIP_OMNIRETARGET_SAVE_INTERVAL:-4000}"

if [[ ! -x "$HOLOSOMA_TRAIN_PYTHON" ]]; then
    echo "Training Python is not executable: $HOLOSOMA_TRAIN_PYTHON" >&2
    exit 1
fi

cd "$HOLOSOMA_ROOT"
export OMNI_KIT_ACCEPT_EULA=1

exec "$HOLOSOMA_TRAIN_PYTHON" src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-wbt-fast-sac-w-object \
    logger:disabled \
    --training.headless=True \
    --training.num-envs="$OMNIRETARGET_NUM_ENVS" \
    --algo.config.num-learning-iterations="$OMNIRETARGET_TRAIN_STEPS" \
    --algo.config.save-interval="$OMNIRETARGET_SAVE_INTERVAL" \
    --algo.config.compile=False \
    "$@"
