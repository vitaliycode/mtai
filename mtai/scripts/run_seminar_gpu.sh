#!/usr/bin/env bash
# Execute seminar notebooks on GPU (conda torch 2.5.1+cu124).
#
# Required: PYTHONNOUSERSITE=1 (ignore ~/.local torch 2.12+cu130 — breaks CUDA on this host).
# Outputs: seminar_*/<name>_gpu_executed.ipynb  Logs: logs/seminar_*_gpu.log
# Seminar 4 first run trains baseline if MIT checkpoint is unreachable (~20–30 min on H100).
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONNOUSERSITE=1
export SEMINAR_DEPS="${REPO}/.seminar_deps"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON=/opt/miniconda3/bin/python
export PATH="/opt/miniconda3/bin:$PATH"

echo "=== GPU check ==="
$PYTHON -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

mkdir -p "${SEMINAR_DEPS}"
$PYTHON "${REPO}/scripts/patch_seminars_gpu.py"

run_nb() {
  local name="$1"
  local nb="$2"
  local log="${REPO}/logs/${name}_gpu.log"
  local out="${nb%.ipynb}_gpu_executed.ipynb"
  mkdir -p "${REPO}/logs"
  echo "=== Running ${name} on GPU (log: ${log}) ==="
  local nb_dir
  nb_dir="$(dirname "${nb}")"
  local nb_base
  nb_base="$(basename "${nb}" .ipynb)"
  $PYTHON -m jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=7200 \
    --ExecutePreprocessor.kernel_name=python3 \
    "${nb}" \
    --output-dir "${nb_dir}" \
    --output "${nb_base}_gpu_executed.ipynb" \
    2>&1 | tee "${log}"
}

case "${1:-all}" in
  4) run_nb "seminar_4" "${REPO}/seminar_4/Pruning.ipynb" ;;
  5) run_nb "seminar_5" "${REPO}/seminar_5/Post_Training_Quantization.ipynb" ;;
  all)
    run_nb "seminar_4" "${REPO}/seminar_4/Pruning.ipynb"
    run_nb "seminar_5" "${REPO}/seminar_5/Post_Training_Quantization.ipynb"
    ;;
  *) echo "Usage: $0 [4|5|all]"; exit 1 ;;
esac

echo "Done."
