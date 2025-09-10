#!/usr/bin/env bash
# my_script.sh
# - Retries until success (with exponential backoff)
# - Uses +trainer.val_before_train=True only on the very first run
#   (later runs + all retries switch it to False)
# To force a "first run" again: rm -f "$HOME/.cache/verl_run_state/<project>__<experiment>.first_run_done"

set -xu -o pipefail

ulimit -n 1048576
# export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1000000000

# -------- Parse command line arguments (supports: --model <PATH>) --------
MODEL_PATH=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL_PATH="$2"
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

# -------- Defaults --------
if [[ -z "${MODEL_PATH:-}" ]]; then
    MODEL_PATH="/mnt/task_wrapper/40-400-qwen-10warmup-5penalty-log-005lenpenalty-3sync_step2400"
fi

PROJECT='rlvr-sep10'
EXPERIMENT='rl-40-400-qwen-10warmup-5penalty-log-005lenpenalty-3sync_step2400-areal'

# -------- First-run marker controls val_before_train --------
STATE_DIR="$HOME/.cache/verl_run_state"
rm -r "$STATE_DIR"
mkdir -p "$STATE_DIR"
STATE_FILE="$STATE_DIR/${PROJECT}__${EXPERIMENT}.first_run_done"

if [[ -f "$STATE_FILE" ]]; then
  VAL_FLAG="trainer.val_before_train=False"
else
  VAL_FLAG="trainer.val_before_train=True"
  : > "$STATE_FILE"   # mark now so any retry or future invocation flips to False
fi

# -------- Retry loop (until success) --------
attempt=1
delay=5            # seconds; will back off up to 300s

#while true; do
python3 -m verl.trainer.main_ppo \
algorithm.adv_estimator=grpo \
data.train_files="$HOME/rllm/data/deepcoder_train.parquet" \
data.val_files="[$HOME/rllm/data/test_codeforces.parquet,$HOME/rllm/data/test_livecodebench.parquet,$HOME/rllm/data/val_areal.parquet]" \
data.train_batch_size=128 \
data.val_batch_size=512 \
data.max_prompt_length=2048 \
data.max_response_length=800 \
actor_rollout_ref.model.path="$MODEL_PATH" \
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
actor_rollout_ref.rollout.val_kwargs.do_sample=False \
actor_rollout_ref.rollout.val_kwargs.temperature=0.00 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
actor_rollout_ref.rollout.n=4 \
actor_rollout_ref.ref.fsdp_config.param_offload=True \
algorithm.kl_ctrl.kl_coef=0.001 \
trainer.critic_warmup=0 \
trainer.logger="['console','wandb']" \
trainer.project_name="$PROJECT" \
trainer.experiment_name="$EXPERIMENT" \
"$VAL_FLAG" \
trainer.n_gpus_per_node=8 \
trainer.nnodes=1 \
trainer.save_freq=10 \
trainer.test_freq=10 \
trainer.default_hdfs_dir=null \
trainer.total_epochs=100 \
"$@"

  rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "Training completed successfully."
    break
  fi

  echo "Attempt $attempt failed with exit code $rc. Retrying in ${delay}s..." >&2
  VAL_FLAG="+trainer.val_before_train=False"         # ensure all retries skip pre-train val
  sleep "$delay"
  ((attempt++))
done
