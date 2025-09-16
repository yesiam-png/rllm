#!/usr/bin/env bash
# Launch parallel evals for checkpoints 10..300 (by 10) and lengths {1024, 2048}.
# - Runs two val modes per task:
#     1) deterministic: do_sample=False, temperature=0.00, n=1
#     2) sample5:      do_sample=True,  temperature=0.6,  n=5
# - Uses 8 workers, each pinned to a different GPU (0..7)
# - Writes logs under ./eval_logs/<timestamp>/L{len}/{mode}/gs_{step}.log
# - After runs, calls parse_eval_logs.py to print a summary table (+ CSV)

set -Eeuo pipefail

##############################
# ---- USER CONFIG HERE ---- #
##############################

# Baseline path from your script (parent dir that contains global_step_10, _20, ... _300)
DEFAULT_MODEL_PATH="/mnt/task_wrapper/rl-40-400-qwen-10warmup-5penalty-log-005lenpenalty-3sync_step2400-areal-1024-temp0/global_step_10"
CHECKPOINT_PARENT="$(dirname "$DEFAULT_MODEL_PATH")"

PROJECT='rlvr-sep12-evalonly'
EXPERIMENT='rl-40-400-qwen-10warmup-5penalty-log-005lenpenalty-3sync_step2400-areal-1024-temp0'

# Prompt-length settings to test (we also mirror response_length to match your prior script)
LENGTHS=(1024 2048)

# Evaluate these checkpoints:
START=0
END=320
STEP=10

# GPUs: one job per GPU, 8 workers total
NUM_WORKERS=8   # expects GPUs 0..7

# Extra args (optional) appended verbatim to the Python call (e.g., override val files)
EXTRA_ARGS=()

# Validation modes
VAL_MODES=(deterministic sample5)

##############################
# ---- ENV & PREP ---------  #
##############################

ulimit -n 1048576 || true
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

timestamp="$(date +%Y%m%d-%H%M%S)"
LOG_ROOT="$(pwd)/eval_logs/${timestamp}"
mkdir -p "${LOG_ROOT}"

echo "[INFO] Logs -> ${LOG_ROOT}"
echo "[INFO] Checkpoint parent: ${CHECKPOINT_PARENT}"
echo "[INFO] Project/Experiment: ${PROJECT} / ${EXPERIMENT}"
echo "[INFO] GPUs/workers: ${NUM_WORKERS} (expects CUDA_VISIBLE_DEVICES 0..$((NUM_WORKERS-1)))"
echo "[INFO] Val modes: ${VAL_MODES[*]}"
echo

##############################
# ---- WORKER FUNCTION ----- #
##############################

run_one() {
  local gpu="$1" step="$2" L="$3" mode="$4"

  export CUDA_VISIBLE_DEVICES="${gpu}"

  local model_path="${CHECKPOINT_PARENT}/global_step_${step}"
  local log_dir="${LOG_ROOT}/L${L}/${mode}"
  local log_file="${log_dir}/gs_${step}.log"
  mkdir -p "${log_dir}"

  # Map mode -> val_kwargs
  local VK_DO_SAMPLE VK_TEMP VK_N
  case "${mode}" in
    deterministic) VK_DO_SAMPLE=False; VK_TEMP=0.00; VK_N=1 ;;
    sample5)       VK_DO_SAMPLE=True;  VK_TEMP=0.6;  VK_N=5 ;;
    *) echo "[GPU ${gpu}] Unknown mode: ${mode}" >&2; return 2 ;;
  esac

  echo "[GPU ${gpu}] step=${step}, L=${L}, mode=${mode} (do_sample=${VK_DO_SAMPLE}, T=${VK_TEMP}, n=${VK_N})"

  # --- Eval-only run (mirrors your hyperparams; val_before_train=True, total_epochs=0) ---
  # Note: We set both max_prompt_length and max_response_length to L to match your prior script.
  # If you intend to only change prompt length, remove the response_length line.
  python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$HOME/rllm/data/deepcoder_train.parquet" \
    data.val_files="[$HOME/rllm/data/test_codeforces.parquet,$HOME/rllm/data/test_livecodebench.parquet,$HOME/rllm/data/val_areal.parquet]" \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length="${L}" \
    data.max_response_length=1024 \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=16 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32000 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=${VK_DO_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${VK_TEMP} \
    actor_rollout_ref.rollout.val_kwargs.n=${VK_N} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger="['console']" \
    trainer.project_name="${PROJECT}" \
    trainer.experiment_name="${EXPERIMENT}-eval-gs${step}-L${L}-${mode}" \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=0 \
    trainer.test_freq=1 \
    trainer.default_hdfs_dir=null \
    trainer.total_epochs=0 \
    "${EXTRA_ARGS[@]}" \
    > "${log_file}" 2>&1

  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[GPU ${gpu}] ✅ Done step=${step}, L=${L}, mode=${mode}"
  else
    echo "[GPU ${gpu}] ❌ FAILED step=${step}, L=${L}, mode=${mode} (rc=${rc}) — see ${log_file}" >&2
  fi
  return $rc
}

export -f run_one
export CHECKPOINT_PARENT LOG_ROOT PROJECT EXPERIMENT EXTRA_ARGS
export VLLM_ENGINE_ITERATION_TIMEOUT_S

##############################
# ---- BUILD TASK QUEUE ---- #
##############################

TASKS_FILE="$(mktemp)"
trap 'rm -f "$TASKS_FILE"' EXIT

for L in "${LENGTHS[@]}"; do
  for s in $(seq "${START}" "${STEP}" "${END}"); do
    for mode in "${VAL_MODES[@]}"; do
      echo "${s} ${L} ${mode}" >> "${TASKS_FILE}"
    done
  done
done

TOTAL_TASKS=$(wc -l < "${TASKS_FILE}")
echo "[INFO] Total tasks: ${TOTAL_TASKS}"
echo


##############################
# ---- START WORKERS ------- #
##############################

# Token queue: one token per GPU id (0..NUM_WORKERS-1)
GPUQ="$(mktemp -u)"
mkfifo "$GPUQ"
# Clean up FIFO and task file on exit
trap 'exec 3>&- 3<&- 2>/dev/null; rm -f "$GPUQ" "$TASKS_FILE"' EXIT

# Open the FIFO for both reading and writing in the parent so writers never block
exec 3<>"$GPUQ"

# Seed tokens (one per GPU)
for (( g=0; g<NUM_WORKERS; g++ )); do
  printf '%s\n' "$g" >&3
done

pids=()
failures=0

# Each task line: "<step> <L> <mode>"
while IFS=' ' read -r step L mode; do
  [[ -z "${step:-}" || -z "${L:-}" || -z "${mode:-}" ]] && continue

  # Wait for an available GPU token
  read -r gpu <&3

  (
    # run_one takes care of export CUDA_VISIBLE_DEVICES="$gpu"
    run_one "$gpu" "$step" "$L" "$mode"
    rc=$?
    # Return the GPU token
    printf '%s\n' "$gpu" >&3
    exit "$rc"
  ) &
  pids+=($!)
done < "${TASKS_FILE}"

# Wait for all jobs
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures+1))
  fi
done

echo
echo "[INFO] All tasks finished. Failures: ${failures}"

# Close FIFO FDs before removal (handled by trap too)
exec 3>&- 3<&-

##############################
# ---- SUMMARIZE RESULTS --- #
##############################
python3 -u "$(dirname "$0")/parse_eval_logs.py" "${LOG_ROOT}" || {
  echo "[WARN] Could not parse logs automatically. Logs are under: ${LOG_ROOT}"
}
