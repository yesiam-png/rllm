#!/usr/bin/env bash
set -euo pipefail

SRC_BASE="/mnt/task_wrapper/user_output/artifacts/checkpoints/rlvr-sep6/rl-qwen15b-temp001-areal"
DST_BASE="s3://afm-common-permanent/shenao_zhang/rl-qwen15b-temp001-areal"

for step in $(seq 10 10 300); do
  src="${SRC_BASE}/actor/global_step_${step}/"
  dst="${DST_BASE}/global_step_${step}"
  echo "Uploading ${src} -> ${dst}"
  if [[ -d "$src" ]]; then
    aws s3 cp "$src" "$dst" --recursive
  else
    echo "WARNING: missing ${src}, skipping." >&2
  fi
done
