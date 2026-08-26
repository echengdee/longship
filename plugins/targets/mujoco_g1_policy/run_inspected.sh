#!/usr/bin/env bash

set -euo pipefail

plugin_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
longship_root="$(cd -- "${plugin_dir}/../../.." && pwd)"

: "${UNITREE_RL_GYM_ROOT:?set UNITREE_RL_GYM_ROOT to the external provider}"
g1_python="${LONGSHIP_G1_PYTHON:-python3}"
codex_sdk_path="${LONGSHIP_CODEX_SDK_PATH:-${longship_root}/.runtime/codex-sdk}"
runtime_pythonpath="${longship_root}/src"
if [[ -d "${codex_sdk_path}" ]]; then
  runtime_pythonpath="${runtime_pythonpath}:${codex_sdk_path}"
fi

exec env PYTHONPATH="${runtime_pythonpath}${PYTHONPATH:+:${PYTHONPATH}}" \
  "${g1_python}" "${plugin_dir}/runner.py" \
  --profile "${longship_root}/scenarios/follow_person/profile.v0.json" \
  --scenario "${longship_root}/scenarios/follow_person/closed_loop.v0.json" \
  --scene "${UNITREE_RL_GYM_ROOT}/resources/robots/g1_description/scene.xml" \
  --scene-bundle-root \
  "${UNITREE_RL_GYM_ROOT}/resources/robots/g1_description" \
  --policy "${UNITREE_RL_GYM_ROOT}/deploy/pre_train/g1/motion.pt" \
  --policy-config \
  "${UNITREE_RL_GYM_ROOT}/deploy/deploy_mujoco/configs/g1.yaml" \
  --license "${UNITREE_RL_GYM_ROOT}/LICENSE" \
  --expected-scene-bundle-sha256 \
  f569b1425fc055ca759699f36f94eba97663db547b79e663bafa50560a0c9349 \
  --expected-policy-sha256 \
  cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d \
  --expected-config-sha256 \
  73044e7d355c61915695c16d6e09eb3efef46eec1e3d708fd3eb9157dfe3bbbb \
  --expected-license-sha256 \
  aef6394ba1597725a68308167324e675f562e6606027404deb1b9da254c2b9c1 \
  "$@"
