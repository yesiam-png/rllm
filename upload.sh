#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/mnt/task_wrapper/user_output/artifacts/checkpoints/rlvr-sep13/rl-60-400-codegemma-10warmup-nopenalty-log-002lenpenalty-2sync_step900-2048-areal-temp0"
DST_BASE="s3://afm-common-permanent/shenao_zhang/rl-60-400-codegemma-10warmup-nopenalty-log-002lenpenalty-2sync_step900-2048-areal-temp0"

for step in $(seq 10 10 80); do
  src="${SRC_BASE}/global_step_${step}/actor/huggingface"
  dst="${DST_BASE}/global_step_${step}"
  echo "Uploading ${src} -> ${dst}"
  if [[ -d "$src" ]]; then
    aws s3 cp "$src" "$dst" --recursive
  else
    echo "WARNING: missing ${src}, skipping." >&2
  fi
done
