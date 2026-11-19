#!/usr/bin/env bash
set -euo pipefail
prep_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$prep_dir/../../.." && pwd)
data_root=/local/data/shared/huggingface_cache/harbor/terminal-bench-pro-82bc46de
export HF_HOME=/local/data/shared/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export STORAGE_ROOT="${STORAGE_ROOT:-/local2/data/jc6183/skyrl_runs/tbpro-qwen35-9b-async50}"
export WANDB_DIR="$STORAGE_ROOT/logs"
export WANDB_MODE=online

# Use task.toml time limits; StateFork inherits the agent limit when no search limit is supplied.
cd "$repo_root"
exec bash examples/train_integrations/harbor/run_codecontest_fully_async.sh \
  "data.train_data=['$data_root']" \
  trainer.policy.model.path=Qwen/Qwen3.5-9B \
  generator.inference_engine.served_model_name=Qwen3.5-9B \
  trainer.policy.language_model_only=true \
  trainer.ref.language_model_only=true \
  generator.inference_engine.language_model_only=true \
  generator.inference_engine.gpu_memory_utilization=0.85 \
  "generator.inference_engine.engine_init_kwargs.chat_template=$repo_root/skyrl/train/utils/templates/qwen35_acc_thinking.jinja2" \
  trainer.max_training_steps=50 trainer.epochs=5 trainer.resume_mode=latest \
  trainer.run_name=tbpro-qwen35-9b-async50 \
  "trainer.ckpt_path=$STORAGE_ROOT/ckpts" \
  "trainer.export_path=$STORAGE_ROOT/exports" \
  "trainer.log_path=$STORAGE_ROOT/logs" \
  "harbor_trial_config.trials_dir=$STORAGE_ROOT/trials" \
  trainer.eval_before_train=false trainer.eval_interval=0 'data.val_data=[]' \
  trainer.ckpt_interval=5 trainer.max_ckpts_to_keep=1 trainer.hf_save_interval=50 \
  trainer.algorithm.zero_variance_filter=true \
  trainer.fully_async.sample_full_batch=true \
  trainer.algorithm.eps_clip_low=0.2 \
  trainer.algorithm.eps_clip_high=0.28 \
  trainer.max_tokens_per_microbatch=32768 \
  trainer.remove_microbatch_padding=true \
  generator.inference_engine.router_init_kwargs.port=18080 \
  generator.inference_engine.enable_ray_prometheus_stats=false \
  generator.remote.endpoint=http://127.0.0.1:18081 \
  generator.remote.inference_url=http://127.0.0.1:18082/v1 \
  "generator.remote.local_task_root=$data_root" \
  generator.remote.remote_task_root=/mydata/local/harbor-data/Terminal-Bench-Pro \
  harbor_trial_config.environment.type=waypoint \
  harbor_trial_config.environment.kwargs.detach_stdin=true \
  harbor_trial_config.environment.kwargs.apply_image_entrypoint=true \
  harbor_trial_config.environment.kwargs.waypoint_network=private \
  harbor_trial_config.environment.kwargs.waypoint_total_memory_mb=98304 \
  harbor_trial_config.environment.kwargs.waypoint_bin=/mydata/local/waypoint/bin/waypoint \
  harbor_trial_config.environment.kwargs.waypoint_bash_init=/mydata/local/waypoint/bin/bash_init \
  harbor_trial_config.environment.kwargs.waypoint_sessions_dir=/mydata/local/wp-sessions \
  'harbor_trial_config.environment.kwargs.extra_docker_compose=[]' \
  harbor_trial_config.agent.override_timeout_sec=null \
  harbor_trial_config.agent.kwargs.execution_backend=headless \
  harbor_trial_config.search.enabled=true \
  harbor_trial_config.search.navigator.name=best_of_n \
  harbor_trial_config.search.navigator.kwargs.n=1 \
  harbor_trial_config.search.navigator.kwargs.parallel=false \
  harbor_trial_config.search.verification_policy.name=single_submit \
  harbor_trial_config.search.verify_on_fork=false \
  "$@"
