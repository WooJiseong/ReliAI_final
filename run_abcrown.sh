#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export XDG_DATA_HOME="${ROOT_DIR}/.cache/xdg"
export MPLCONFIGDIR="${ROOT_DIR}/.cache/matplotlib"

mkdir -p "${ROOT_DIR}/verification/results" "${XDG_DATA_HOME}" "${MPLCONFIGDIR}"

"${PYTHON:-python}" "${ROOT_DIR}/verification/create_fixture.py" \
  --checkpoint "${ABCROWN_SOURCE_CHECKPOINT:-model/AudioCResNet5.pt}" \
  --abcrown-checkpoint "model/AudioCResNet5_abcrown.pt" \
  --count "${ABCROWN_SAMPLE_COUNT:-20}" \
  --sample-strategy "${ABCROWN_SAMPLE_STRATEGY:-random}"

cd "${ROOT_DIR}/alpha-beta-CROWN/complete_verifier"
../.venv/bin/python abcrown.py --config "${ROOT_DIR}/verification/audio_cresnet5.yaml"

cd "${ROOT_DIR}"
"${PYTHON:-python}" "${ROOT_DIR}/verification/export_safe_samples.py"
"${PYTHON:-python}" "${ROOT_DIR}/verification/export_report_assets.py"
