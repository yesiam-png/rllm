#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/mnt/task_wrapper/user_output/artifacts/checkpoints/rlvr-sep21/rl-llama31-8b-2048-areal-temp0-naive-run1"
DST_BASE="s3://afm-common-permanent/shenao_zhang/rl-llama31-8b-2048-areal-temp0-naive-run1"

for step in $(seq 10 10 90); do
  src="${SRC_BASE}/global_step_${step}/actor/huggingface"
  dst="${DST_BASE}/global_step_${step}"
  echo "Uploading ${src} -> ${dst}"
  if [[ -d "$src" ]]; then
    aws s3 cp "$src" "$dst" --recursive
  else
    echo "WARNING: missing ${src}, skipping." >&2
  fi
done
