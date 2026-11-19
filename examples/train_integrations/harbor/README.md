## Harbor Integration

RL training with [Harbor](https://github.com/laude-institute/harbor) as the environment and reward source. See the [full documentation](https://docs.skyrl.ai/docs/harbor) for details.

### Structure

```
skyrl/integration/harbor/
  harbor_generator.py              # HarborGenerator: bridges SkyRL <-> Harbor
  forkrl_generator.py              # Group-based submit/poll/ack transport
  forkrl_trainer.py                # Optional remote per-turn advantages (sync/async)
  dataset.py                       # HarborTaskDataset: loads task directory paths
  trial_metrics.py                 # Per-step trial phase and retry metrics

examples/train_integrations/harbor/
  prepare_harbor_dataset.py        # Downloads + extracts datasets from HuggingFace
  harbor_trial_config/
    default.yaml                   # Harbor TrialConfig template
  entrypoints/
    main_harbor.py                 # Full training entrypoint
    main_harbor_generate.py        # Generation-only debug entrypoint
  run_codecontest.sh               # Code contest training (Qwen3-8B)
  run_harbor_gen.sh                # Debug generation-only
```

### Quick Start

```bash
cd SkyRL

# 1. Set credentials
export WANDB_API_KEY=your_wandb_api_key
# Pick your sandbox provider:
export DAYTONA_API_KEY=your_daytona_api_key
# export MODAL_TOKEN_ID=your_modal_token_id
# export MODAL_TOKEN_SECRET=your_modal_token_secret

# 2. Prepare dataset
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/CodeContests
uv run examples/train_integrations/harbor/prepare_harbor_dataset.py \
    --dataset open-thoughts/OpenThoughts-TB-dev

# 3. Launch training
bash examples/train_integrations/harbor/run_codecontest.sh
```

### StateFork on a separate CPU host

The optional remote transport keeps training, vLLM, and weight synchronization on
the GPU host. A CPU host runs the patched `harbor-StateFork` checkout, the agent,
search, sandboxes, and verifier. The GPU host can keep its existing Harbor version.
The two source checkouts must include this adapter and the rollout-server/provenance
changes together; the unmodified StateFork checkout does not expose this endpoint.

First prepare Waypoint on the CPU host using its
[`src/harbor/environments/waypoint/README.md`](https://github.com/Alex-XJK/harbor-StateFork/blob/develop/src/harbor/environments/waypoint/README.md).
This includes StateFork, the Waypoint binaries, buildah, CRIU, passwordless sudo,
and a disk-backed sessions directory. `uv sync` alone does not install these.
Copy the task directories to the CPU host, dereferencing HF-cache symlinks (for
example, `rsync -aL`). Both hosts need task metadata; only the GPU host needs weights.

On the CPU host, from the patched Harbor checkout and with caches/venvs located on
its data disk, start the service:

```bash
uv run --isolated -m harbor.rollout_server \
  --trials-dir /srv/runs/skyrl-statefork/trials \
  --max-concurrency 512
```

The service binds to `127.0.0.1:18081`. Its trial configs execute code: keep it on
loopback and access it through SSH. On the GPU host, open both tunnels in a separate
terminal, replacing `CPU_HOST` with the SSH host:

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -L 127.0.0.1:18081:127.0.0.1:18081 \
  -R 127.0.0.1:18080:127.0.0.1:18080 CPU_HOST
```

`18081` carries trial requests/results from GPU to CPU. `18080` carries agent model
requests from CPU back to SkyRL's router. Reserve a free router port: SkyRL may
choose the next port if `18080` is occupied, so its logged router port must match
the tunnel destination. This example assumes both SkyRL's driver and router are on
the same GPU host, as in the single-node training script.

With the existing training environment activated on the GPU host, a one-step
run can use the following overrides: 32 tasks with 8 samples each (256 rollouts),
with a concurrency limit of 512 on both hosts. Replace `/GPU/DATA/harbor` with
the dataset root used locally and `/srv/data/harbor` with its CPU-host copy. The
local root may be a symlink; it is resolved on the GPU host. Use a fresh output
directory with enough space for the script's final checkpoint/export.

```bash
STORAGE_ROOT=/local2/data/$USER/skyrl_runs/statefork-smoke \
bash examples/train_integrations/harbor/run_codecontest.sh \
  trainer.max_training_steps=1 \
  trainer.resume_mode=null \
  trainer.eval_interval=0 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=32 \
  generator.n_samples_per_prompt=8 \
  generator.rate_limit.max_concurrency=512 \
  generator.remote.endpoint=http://127.0.0.1:18081 \
  generator.remote.inference_url=http://127.0.0.1:18080/v1 \
  generator.remote.local_task_root=/GPU/DATA/harbor \
  generator.remote.remote_task_root=/srv/data/harbor \
  generator.inference_engine.router_init_kwargs.port=18080 \
  harbor_trial_config.environment.type=waypoint \
  'harbor_trial_config.environment.kwargs.extra_docker_compose=[]' \
  harbor_trial_config.agent.kwargs.execution_backend=headless \
  harbor_trial_config.environment.kwargs.detach_stdin=true \
  harbor_trial_config.agent.kwargs.enable_summarize=false \
  harbor_trial_config.agent.kwargs.collect_rollout_details=true \
  harbor_trial_config.search.enabled=true \
  harbor_trial_config.search.navigator.name=best_of_n \
  harbor_trial_config.search.navigator.kwargs.n=1 \
  harbor_trial_config.search.navigator.kwargs.parallel=false \
  harbor_trial_config.search.verification_policy.name=single_submit \
  harbor_trial_config.search.verify_on_fork=false
```

This initial check exercises the search path with one attempt per search. Each
search contributes its submitted conversation and matching verifier reward to the
existing SkyRL prompt/repetition group. The adapter checks node identity and reward;
SkyRL's existing output conversion checks token/logprob shapes. Summarized or
multiple-segment submissions are rejected. The
advanced parallel-search and LLM-critic configurations are not covered by this
connection check.

One HTTP request runs one trial and returns its result. In this legacy trial transport,
agent timeouts and startup errors mask the prompt group. Context-length errors,
exhausted searches, and other reported environment errors mask only that trajectory.
Reported trial failures are not retried; connection and result-contract errors propagate.
On request timeout/cancellation the client closes its request;
the running CPU service detects the disconnect and cancels that trial. This relies
on the service staying alive to perform cleanup. Trial files remain on the CPU host.

CPU tests cover the HTTP service with a fake trial and the search export through
SkyRL's training-output conversion. A real Waypoint task and a GPU training step
still need to be checked on the intended two machines.

### ForkRL per-turn advantages

The ForkRL group service uses `generator.remote.protocol=forkrl`. By default,
SkyRL computes ordinary GRPO/DAPO advantages from the returned rewards.
The adapter accepts current `trajectory_id` results in group order and older
results with explicit `index` fields, keeping each rollout's tokens, reward and
per-turn credit together. Mixed ID formats, duplicate IDs and wrong group sizes
are rejected.
Infrastructure failures carry `reward_valid=false`: their placeholder zeros are
excluded from GRPO group statistics and dynamic sampling. Scored model failures
retain their zero rewards, independently of token loss masking. Groups with fewer
than two valid outcomes receive zero GRPO advantages and are skipped by dynamic sampling.
Requests include `run_id=trainer.run_name` and the trainer's computed
`total_steps` for the ForkRL monitor. Keep the run name when resuming and use a
different name for a new experiment. Both synchronous and fully-async training
send the total planned step count, including the epoch count and step cap;
generation without a trainer omits `total_steps`. These fields require a service
version that supports the monitor protocol. The async trainer may exhaust its
dataset before the planned total when filtering groups, so the monitor's progress
is not a guarantee of how many updates will run.
Set `generator.remote.max_turns=50` to override the CPU service's turn limit for
each rollout in this run. The default `null` sends no override. The CPU service
must support the optional `max_turns` request field; subsequent runs can change
this value without restarting it. Context and time limits remain independent.

To use the service's `turns[].advantage` instead, add these overrides to the
existing recipe (the same switch works with the synchronous and fully-async entrypoints):

```bash
generator.remote.protocol=forkrl \
generator.remote.endpoint=http://127.0.0.1:18083 \
generator.remote.use_step_advantages=true \
generator.step_wise_trajectories=true \
generator.merge_stepwise_output=false \
trainer.algorithm.zero_variance_filter=false \
trainer.fully_async.sample_full_batch=false \
trainer.algorithm.advantage_batch_normalize=false \
trainer.algorithm.use_kl_in_reward=false \
trainer.critic.model.path=null
```

The generator preserves one advantage per turn and supplies zero for masked rows.
`ForkRLAdvantagesMixin` carries these values through tensor conversion and batch
padding, then broadcasts each value only to that turn's valid response tokens.
SkyRL's reward-based estimator is bypassed; native loss reduction, clipping,
off-policy correction, optimizer, weight synchronization and async scheduling remain.
Reward metrics still use the verifier rewards, independently of the supplied advantages.

The incompatible settings above are checked before starting Ray. Turn merging
would combine rows with different credit; reward-only zero-variance filtering
could drop a group whose per-turn credit is nonzero. Reward KL and batch advantage
normalization would change or ignore the externally assigned training signal.
Critic training is not supported by this path. No settings are silently overridden.

`fake-demo` in forkrl-pipeline can exercise submit/poll/ack and blame credit without
real environments; use at least four rollouts for its default requirement of two
successful rollouts. This tests the GPU-side data path, not the live StateFork
diff/blame implementation. Ordinary GRPO/DAPO runs keep `use_step_advantages=false`.
