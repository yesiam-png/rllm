#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/mnt/task_wrapper/user_output/artifacts/checkpoints/rlvr-sep12/rl-40-400-qwen-10warmup-5penalty-log-005lenpenalty-3sync_step2400-areal-1024-temp0"
DST_BASE="s3://afm-common-permanent/shenao_zhang/rl-40-400-qwen-10warmup-5penalty-log-005lenpenalty-3sync_step2400-areal-1024-temp0"

for step in $(seq 10 10 300); do
  src="${SRC_BASE}/global_step_${step}/actor/huggingface"
  dst="${DST_BASE}/global_step_${step}"
  echo "Uploading ${src} -> ${dst}"
  if [[ -d "$src" ]]; then
    aws s3 cp "$src" "$dst" --recursive
  else
    echo "WARNING: missing ${src}, skipping." >&2
  fi
done
