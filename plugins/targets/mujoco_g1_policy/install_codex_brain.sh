#!/usr/bin/env bash

set -euo pipefail

plugin_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
longship_root="$(cd -- "${plugin_dir}/../../.." && pwd)"
g1_python="${LONGSHIP_G1_PYTHON:-python3}"
sdk_path="${LONGSHIP_CODEX_SDK_PATH:-${longship_root}/.runtime/codex-sdk}"

command -v uv >/dev/null 2>&1 || {
  printf 'BLOCKED: uv is required to install the optional Codex SDK\n' >&2
  exit 3
}
[[ -x "${g1_python}" ]] || command -v "${g1_python}" >/dev/null 2>&1 || {
  printf 'BLOCKED: G1 Python is not executable: %s\n' "${g1_python}" >&2
  exit 3
}

mkdir -p "${sdk_path}"
uv pip install \
  --python "${g1_python}" \
  --target "${sdk_path}" \
  'openai-codex>=0.144.4,<0.145'

printf 'Codex Brain SDK installed at %s\n' "${sdk_path}"
printf 'The SDK reuses the current Codex login; verify it with: codex login status\n'
