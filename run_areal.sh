#!/usr/bin/env bash
# runner.sh
set -u

until ./scripts/deepcoder/train/deepcoder_14b_coding_16k.sh; do
  rc=$?
  echo "Run failed with exit code $rc — retrying in 5s..." >&2
  sleep 4
done

echo "Completed successfully."

