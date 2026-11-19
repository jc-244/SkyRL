# Tinker Integration for SkyRL Agent

This directory contains the integration between SkyRL Agent and Tinker's RL training framework.

## Overview

The Tinker integration enables you to:
1. Use **AgentRunner** to collect trajectories from your agents
2. Train models using **Tinker's LoRA training** with importance sampling or PPO
3. Track metrics with **Weights & Biases**
4. Save and resume from checkpoints

## Architecture

### Components

1. **`tinker_backend.py`**: Backend interface for Tinker's sampling client
   - Handles async generation from token IDs
   - Decodes outputs with proper special token handling
   - Detects finish reasons (EOS vs length limit)

2. **`tinker_train.py`**: Main training script
   - Loads data from Parquet files
   - Runs AgentRunner for trajectory collection
   - Converts AgentRunner output to Tinker's format
   - Performs RL training with advantage normalization
   - Saves checkpoints periodically

3. **`__init__.py`**: Backend registration
   - Registers Tinker backend in the SkyRL Agent registry

## Data Flow

```
Input Batch (from Parquet)
  ↓
AgentRunner.run() → Trajectories
  ↓
Extract: {prompt_token_ids, response_ids, logprobs, rewards, loss_masks}
  ↓
Compute: Advantages (with optional normalization)
  ↓
Prepare: Tinker Datums (with masks and advantages)
  ↓
Tinker Training: forward_backward() + optim_step()
  ↓
Metrics & Checkpointing
```

## Vanilla Harbor rollouts

`tinker_train.py` also accepts `harbor_trial_config` instead of
`skyrl_agent_task_yaml`. Harbor runs Terminus-2 in Docker on the CPU host and
uses its built-in Tinker backend for inference. SkyRL retains the existing
GRPO advantages, PPO loss, optimizer, and checkpoint loop. This is synchronous
LoRA training; it does not use the Ray fully asynchronous trainer or StateFork.

Install `harbor[tinker]`, `tinker-cookbook`, `chz`, `torch`, `datasets`, `wandb`,
`transformers`, and `pyyaml` in a separate CPU environment. Docker and Docker
Compose must be usable by the process. No local GPU backend is required.

The training Parquet has one row per question, with `task_path` pointing to its
Harbor task directory on that host. Add `data_source` when using evaluation.
Each training question produces `group_size` independent rollouts; evaluation
produces one. Token IDs and sampling logprobs come directly from Harbor, without
re-tokenizing model text. Each turn trains only its own response tokens.

From the CPU project environment, with `SKYRL_ROOT` set to the checkout path:

```bash
PYTHONPATH="$SKYRL_ROOT" uv run --isolated python \
  "$SKYRL_ROOT/skyrl-agent/skyrl_agent/integrations/tinker/tinker_train.py" \
  harbor_trial_config="$SKYRL_ROOT/examples/train_integrations/harbor/harbor_trial_config/tinker_docker.yaml" \
  model_name=Qwen/Qwen3.5-9B dataset_file=/path/to/train.parquet \
  batch_size=1 group_size=8 harbor_max_concurrency=8 \
  max_steps=1 eval_every=0 wandb_project=skyrl-tinker
```

The example config uses 32 agent turns, a 32,768-token training context,
temperature 1, and Harbor Tinker's default 8,192-token output cap per response.
Task-specific CPU/memory limits remain in effect. Environments are deleted after
each trial; logs are retained in `harbor-trials` and checkpoint references in
`tinker_output`, relative to the working directory. Model weights stay in Tinker.

Ordinary trial errors invalidate their question group. Length-limit failures
exclude only that rollout. Batches with no valid turns skip the optimizer update
and report `rollout/masked_trials`; they are not successful learning steps.

Tested with Harbor 0.15.0, Tinker 0.36.0, and Cookbook 0.5.9. The adapter's
Terminus-2 subclass only converts Cookbook's structured response content to
plain text for Harbor's command parser; original token IDs and logprobs,
including thinking tokens, remain unchanged for training.

## Vanilla Harbor with Cookbook's async trainer

`examples/train_integrations/harbor/tinker_async.py` reuses the same Harbor runner
with the official `tinker_cookbook.rl.train.main` and `AsyncConfig`. It uses
Cookbook's documented `rollouts.do_group_rollout` replacement point; it does not
implement another training loop, sample queue, weight publisher, or staleness
policy. This entrypoint is for hosted Tinker and needs no local GPU.

Each question group receives a sampling client from Cookbook. All turns of that
group retain that client, even while other groups train and publish new weights.
Harbor still runs Terminus-2, creates/deletes Docker environments, grades tasks,
and records the original tokens and logprobs. A shared semaphore limits total
active Harbor trials across all groups. The hook is scoped to the new process;
existing SkyRL/StateFork training services are not changed.

From the separate CPU environment described above:

```bash
PYTHONPATH="$SKYRL_ROOT" uv run --isolated python \
  -m examples.train_integrations.harbor.tinker_async \
  dataset_file=/path/to/train.parquet \
  harbor_trial_config="$SKYRL_ROOT/examples/train_integrations/harbor/harbor_trial_config/tinker_docker.yaml" \
  log_path=/mydata/local/tinker-runs/harbor-async \
  model_name=Qwen/Qwen3.5-9B groups_per_batch=1 group_size=8 \
  harbor_max_concurrency=8 max_steps_off_policy=4 \
  max_steps=2 num_epochs=1 wandb_project=skyrl-tinker
```

`groups_per_batch` is the number of questions per update, `group_size` is the
number of independent rollouts per question, and `harbor_max_concurrency` caps
active trials. `max_steps_off_policy` limits how many model updates a sample may
lag behind. These are separate settings. The defaults above are small integration
settings, not an assertion of optimal throughput. Each epoch has
`floor(number_of_questions / groups_per_batch)` batches; incomplete final batches
are dropped. `max_steps` is also capped by `num_epochs * batches_per_epoch`.

The entrypoint keeps PPO, learning rate `4e-5`, LoRA rank 16, temperature 1, and
8,192 generated tokens per response. The YAML retains the 32-turn and 32,768-token
limits. Evaluation is disabled. Cookbook owns checkpoints, resume, tracing, and
optional W&B logging; reuse the same `log_path` to resume a Cookbook checkpoint.
Synchronous SkyRL-Agent checkpoint metadata is not interchangeable with this
entrypoint's resume metadata.

Two data-processing details follow Cookbook: advantages are centered within each
question group without the synchronous script's extra standard-deviation
normalization, and matching conversation prefixes are merged into training
records with observation tokens masked out. There is no retokenization. Harbor's
ordinary errors still invalidate the question group; length-limit errors remove
only that member's training tokens. A group with no valid tokens is skipped by
Cookbook's existing `AllTrajectoriesFailedError` handling. This integration uses
in-process rollouts, not Cookbook's optional process-pool executor.

Validated with Harbor 0.15.0, Tinker 0.36.0, and Cookbook 0.5.9: eight offline
checks and a three-update Qwen3.5-9B Docker fixture (six successful trials), with
nonzero advantages in one group, measured rollout/update overlap, a newer sampler
used by later rollouts, and a final checkpoint. This was an integration fixture,
not a benchmark evaluation.
