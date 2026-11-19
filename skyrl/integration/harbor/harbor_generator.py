import asyncio
import logging
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, is_dataclass
from typing import List, Optional
from uuid import uuid4

# Suppress LiteLLM verbose logging
import litellm
from harbor.models.agent.rollout_detail import RolloutDetail
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from loguru import logger
from omegaconf import DictConfig
from tqdm import tqdm

from skyrl.backends.skyrl_train.inference_servers.base import (
    ConversationType,
    InferenceEngineInterface,
)
from skyrl.integration.harbor.remote import (
    RemoteHarborConfig,
    map_task_path,
    run_remote_trial,
)
from skyrl.integration.harbor.rollout_adapter import (
    RolloutContractError,
    extract_rollout,
)
from skyrl.integration.harbor.trial_metrics import TrialAttempts, trial_metrics
from skyrl.train.generators.base import (
    GeneratorInput,
    GeneratorInterface,
    GeneratorOutput,
    TrajectoryID,
)
from skyrl.train.generators.utils import build_vllm_cache_salt, get_rollout_metrics
from skyrl.train.utils.rate_limiter import create_rate_limiter

litellm.suppress_debug_info = True  # Suppress the "Provider List" output
litellm.set_verbose = False
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

# We have N retries for each trial, if one of the rollout (out of n_samples_per_prompt) fails
# after N attemptes, we skip this prompt altogether.
MAX_NUM_RETRIES_PER_TRIAL = 2


@dataclass
class HarborTrajectoryOutput:
    """One trajectory's raw output from Harbor.

    Holds the entire ``rollout_details`` from ``agent_result``. Per-step interpretation
    (loss-mask / reward broadcast / overlong filtering) is done downstream in
    `build_step_wise_generator_output`.
    """

    trajectory_id: TrajectoryID
    # Entire rollout_details list as returned by harbor's agent_result. None for failed trajectories
    # (agent_timeout / error) that we will mask in `build_step_wise_generator_output`.
    rollout_details: Optional[List[RolloutDetail]] = None
    reward: float = 0.0
    num_turns: int = 0
    # One of: "complete", "context_length", "search_limits_exhausted", "environment_error", "agent_timeout", "error". Used by
    # `build_step_wise_generator_output` to decide whether to skip the entire prompt group.
    stop_reason: str = "complete"
    # End-to-end wall-clock time (seconds) to generate this trajectory. Optional: left as None if
    # timing was not recorded.
    e2e_time: Optional[float] = None
    # Preserve the remote failure details for evaluation records and diagnosis.
    exception_info: Optional[dict] = None
    # Optional advantages from the remote credit service, one per LLM turn.
    step_advantages: Optional[List[float]] = None


def build_step_wise_generator_output(
    trajectory_outputs: List[HarborTrajectoryOutput], overlong_filtering: bool
) -> GeneratorOutput:
    """Flatten per-trajectory rollout details into one entry per LLM turn.

    Steps for one trajectory are emitted contiguously and the last step has
    ``is_last_step=True``. Failures (timeout / unknown error / empty rollout
    details) are batched per ``instance_id``: if any rollout for prompt P
    failed, all rollouts for P are replaced with single zeroed-out
    placeholder steps. Context-length, search-limit and environment-state failures mask only their own trajectory;
    remote searches without partial tokens use a single masked placeholder.
    """
    use_step_advantages = any(t.step_advantages is not None for t in trajectory_outputs)
    if use_step_advantages and any(t.step_advantages is None for t in trajectory_outputs):
        raise ValueError("Cannot mix remote and reward-based advantages in one batch")
    step_advantages: List[float] = []

    # 1. Identify failed instances. If any rollout for prompt P failed, mask all rollouts for P (conservative).
    timeout_instance_ids = set()
    error_instance_ids = set()
    all_instance_ids = set()
    num_timeout_trajectories = 0
    num_error_trajectories = 0
    for traj in trajectory_outputs:
        instance_id = traj.trajectory_id.instance_id
        all_instance_ids.add(instance_id)
        if traj.stop_reason == "agent_timeout":
            num_timeout_trajectories += 1
            timeout_instance_ids.add(instance_id)
        elif traj.stop_reason == "error" or (
            traj.rollout_details is None
            and traj.stop_reason not in {"context_length", "search_limits_exhausted", "environment_error"}
        ):
            num_error_trajectories += 1
            error_instance_ids.add(instance_id)
    masked_instance_ids = timeout_instance_ids | error_instance_ids

    # 2. Walk trajectories and emit one entry of GeneratorOutput per step.
    prompt_token_ids: List[List[int]] = []
    response_ids: List[List[int]] = []
    rewards: List[float] = []
    reward_valid: List[bool] = []
    loss_masks: List[List[int]] = []
    stop_reasons: List[str] = []
    is_last_step_list: List[bool] = []
    out_trajectory_ids: List[TrajectoryID] = []
    rollout_logprobs_list: List[List[float]] = []

    successful_trajectories: List[HarborTrajectoryOutput] = []
    response_ids_for_metrics: List[List[int]] = []
    rewards_for_metrics: List[float] = []
    # One generation time per successful trajectory; used for completion-time metrics (avoids the
    # duplicate per-step entries below inflating the stats).
    trajectory_generation_times_per_prompt: List[Optional[float]] = []
    # One generation time per emitted step, aligned 1:1 with the flattened per-step arrays above.
    # Per trajectory we replicate its trajectory-level e2e_time across all of its steps.
    out_trajectory_generation_times: List[Optional[float]] = []
    for traj in trajectory_outputs:
        tid = traj.trajectory_id

        # 2.1. Keep a reward slot even when a search returned no partial tokens.
        masked_group = tid.instance_id in masked_instance_ids
        if (
            masked_group
            or traj.stop_reason in {"search_limits_exhausted", "environment_error"}
            or (traj.stop_reason == "context_length" and not traj.rollout_details)
        ):
            prompt_token_ids.append([0])
            response_ids.append([0])
            rewards.append(0.0)
            # Infrastructure failures have no measured outcome. Budget/context stops
            # retain their task-defined zero reward even when no tokens can train.
            reward_valid.append(not masked_group and traj.stop_reason != "environment_error")
            loss_masks.append([0])
            stop_reasons.append("error" if masked_group else traj.stop_reason)
            is_last_step_list.append(True)
            out_trajectory_ids.append(tid)
            rollout_logprobs_list.append([0.0])
            out_trajectory_generation_times.append(traj.e2e_time)
            if use_step_advantages:
                step_advantages.append(0.0)
            continue

        # 2.2. For successful trajectories, emit one entry per step.
        successful_trajectories.append(traj)

        # 2.3. Check rollout_details expected format.
        # Expect no summarization; rollout_details is a single linear chat segment from the main agent.
        # TODO(Charlie): Support summarization.
        assert len(traj.rollout_details) == 1, f"Expected exactly one rollout segment, got {len(traj.rollout_details)}."
        rollout_detail = traj.rollout_details[0]
        prompt_token_ids_per_turn = rollout_detail["prompt_token_ids"]
        completion_token_ids_per_turn = rollout_detail["completion_token_ids"]
        logprobs_per_turn = rollout_detail["logprobs"]
        n_turns = len(completion_token_ids_per_turn)
        if use_step_advantages and len(traj.step_advantages) != n_turns:
            raise ValueError("Remote advantages must match the rollout's turns")
        assert len(prompt_token_ids_per_turn) == n_turns and len(logprobs_per_turn) == n_turns, (
            f"Malformed rollout_details (prompts={len(prompt_token_ids_per_turn)}, completions={n_turns}, "
            f"logprobs={len(logprobs_per_turn)})."
        )

        # 2.4. Emit one entry per step, following SkyRL's step-wise convention.
        for t in range(n_turns):
            comp_ids = completion_token_ids_per_turn[t]
            p_ids = prompt_token_ids_per_turn[t]
            lp = logprobs_per_turn[t]
            assert len(lp) == len(comp_ids), "logprobs and completion token ids must have the same length."

            # Record actual reward in last turn, and zeros for all other turns.
            is_last = t == n_turns - 1
            reward = traj.reward if is_last else 0.0

            # Loss mask.
            step_loss_mask = [1] * len(comp_ids)
            step_stop_reason = "complete"
            if traj.stop_reason == "context_length":
                step_stop_reason = "context_length"
                if overlong_filtering:
                    step_loss_mask = [0] * len(comp_ids)

            prompt_token_ids.append(p_ids)
            response_ids.append(comp_ids)
            rewards.append(reward)
            reward_valid.append(True)
            loss_masks.append(step_loss_mask)
            stop_reasons.append(step_stop_reason)
            is_last_step_list.append(is_last)
            out_trajectory_ids.append(tid)
            rollout_logprobs_list.append(lp)
            # For trajectory completion per turn we just use the trajectory-level e2e time.
            out_trajectory_generation_times.append(traj.e2e_time)
            if use_step_advantages:
                step_advantages.append(traj.step_advantages[t] if any(step_loss_mask) else 0.0)

        # 2.5. For trajectory-level metrics, record the last turn's prompt IDs and response IDs which
        # contains the entire trajectory.
        response_ids_for_metrics.append(prompt_token_ids_per_turn[-1] + completion_token_ids_per_turn[-1])
        rewards_for_metrics.append(traj.reward)
        trajectory_generation_times_per_prompt.append(traj.e2e_time)

    # 3. Aggregate trajectory-level metrics for logging.
    # ``e2e_time`` is optional; if any trajectory did not record it, we omit the completion-time
    # fields entirely rather than emit a partially-populated list.
    # NOTE: metrics use the per-prompt times (not the per-step duplicates) to avoid skewing the stats.
    if any(t is None for t in trajectory_generation_times_per_prompt):
        trajectory_generation_times_per_prompt = None
    if any(t is None for t in out_trajectory_generation_times):
        out_trajectory_generation_times = None
    if successful_trajectories:
        rollout_metrics = get_rollout_metrics(
            response_ids_for_metrics,
            rewards_for_metrics,
            trajectory_completion_times=trajectory_generation_times_per_prompt,
        )
        rollout_metrics["generate/avg_num_turns"] = sum(t.num_turns for t in successful_trajectories) / len(
            successful_trajectories
        )
    else:
        rollout_metrics = {}

    rollout_metrics["generate/num_timeout_trajectories"] = num_timeout_trajectories
    rollout_metrics["generate/num_error_trajectories"] = num_error_trajectories
    rollout_metrics["generate/num_masked_instances"] = len(masked_instance_ids)
    rollout_metrics["generate/trajectories_context_length_exceeded"] = sum(
        t.stop_reason == "context_length" for t in trajectory_outputs
    )
    rollout_metrics["generate/trajectories_search_limits_exhausted"] = sum(
        t.stop_reason == "search_limits_exhausted" for t in trajectory_outputs
    )
    rollout_metrics["generate/trajectories_environment_error"] = sum(
        t.stop_reason == "environment_error" for t in trajectory_outputs
    )

    output = GeneratorOutput(
        prompt_token_ids=prompt_token_ids,
        response_ids=response_ids,
        rewards=rewards,
        reward_valid=reward_valid,
        loss_masks=loss_masks,
        stop_reasons=stop_reasons,
        rollout_metrics=rollout_metrics,
        rollout_logprobs=rollout_logprobs_list,
        trajectory_ids=out_trajectory_ids,
        is_last_step=is_last_step_list,
        # Per-step times, aligned 1:1 with the flattened per-step arrays above.
        trajectory_generation_times=out_trajectory_generation_times,
    )
    if use_step_advantages:
        output["step_advantages"] = step_advantages
    return output


class HarborGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        harbor_cfg: DictConfig,
        inference_engine_client: InferenceEngineInterface,
        tokenizer,
        max_seq_len: int,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            harbor_cfg: DictConfig object containing the Harbor configuration
            inference_engine_client: inference engine client for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
            max_seq_len: Maximum total sequence length (prompt + response). Used to truncate responses.
        """
        ie_cfg = generator_cfg.inference_engine
        if ie_cfg.enable_return_sample_support_set:
            # The agent samples through the OpenAI chat API, which returns no sampler support.
            raise ValueError(
                "HarborGenerator cannot capture sampler support; use the harbor_skycap integration for "
                "sample-support replay, or set generator.inference_engine.enable_return_sample_support_set=false."
            )
        self.base_url = inference_engine_client.get_endpoint_url()
        remote_cfg = getattr(generator_cfg, "remote", None)
        self.remote = (
            RemoteHarborConfig(**(asdict(remote_cfg) if is_dataclass(remote_cfg) else dict(remote_cfg)))
            if remote_cfg is not None
            else RemoteHarborConfig()
        )
        if self.remote.endpoint and not all(
            (
                self.remote.inference_url,
                self.remote.local_task_root,
                self.remote.remote_task_root,
            )
        ):
            raise ValueError("Remote Harbor requires inference_url, local_task_root and remote_task_root")
        # Kept so we can notify the router when a session (trial attempt) ends,
        # which lets session-aware routing policies rebalance new trajectories.
        self.inference_engine_client = inference_engine_client
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        if not getattr(generator_cfg, "step_wise_trajectories", False):
            raise ValueError(
                "HarborGenerator only supports step-wise training. " "Set generator.step_wise_trajectories=true."
            )
        if not getattr(generator_cfg, "merge_stepwise_output", False):
            logger.warning(
                "merge_stepwise_output=true is not set; will not merge step-wise outputs. This "
                "may result in much slower training."
            )

        self._harbor_trial_config_template = deepcopy(harbor_cfg)

        # Mixed into the prefix-cache salt so distinct models / adapters don't share cache blocks.
        self._served_model_name = ie_cfg.served_model_name

        # Set model_name and api_base once (constant across all trials)
        assert ie_cfg.served_model_name is not None, "served_model_name must be set"
        assert (
            "/" not in ie_cfg.served_model_name
        ), "served_model_name must not contain '/', Harbor expects hosted_vllm/{model_name}"
        self._harbor_trial_config_template.setdefault("agent", {})[
            "model_name"
        ] = f"hosted_vllm/{ie_cfg.served_model_name}"
        self._harbor_trial_config_template["agent"].setdefault("kwargs", {})["api_base"] = (
            self.remote.inference_url if self.remote.endpoint else f"{self.base_url}/v1"
        )

        # Step-wise needs per-turn token IDs and logprobs from vLLM via Harbor.
        agent_kwargs = self._harbor_trial_config_template["agent"]["kwargs"]
        if not agent_kwargs.get("collect_rollout_details", False):
            logger.warning("step_wise_trajectories=true requires collect_rollout_details=true; enabling automatically.")
            agent_kwargs["collect_rollout_details"] = True

        # Can support summarization in future.
        if agent_kwargs.get("enable_summarize", False):
            raise ValueError(
                "step_wise_trajectories=true is incompatible with enable_summarize=true. "
                "Set harbor_trial_config.agent.kwargs.enable_summarize=false."
            )

        logger.info(
            f"HarborGenerator initialized with Harbor config. "
            f"Agent: {self._harbor_trial_config_template.get('agent', {}).get('name')}, "
            f"Trials dir: {self._harbor_trial_config_template.get('trials_dir', 'trials')}"
        )

        rate_limit_config = getattr(generator_cfg, "rate_limit", None)
        self._rate_limiter = create_rate_limiter(rate_limit_config)

    def _compute_cache_salt(self) -> Optional[str]:
        """Derive a prefix-cache salt from the current policy version.

        Mirrors ``SkyRLGymGenerator._compute_cache_salt``: keyed on the engine's ``weight_version`` and
        served model name, called once per ``generate`` batch. Returns ``None`` when disabled or when the
        client exposes no weight version.
        """
        if not getattr(self.generator_cfg, "use_cache_salt", False):
            return None
        weight_version = getattr(self.inference_engine_client, "weight_version", None)
        if weight_version is None:
            return None
        return build_vllm_cache_salt(weight_version, self._served_model_name)

    async def generate(self, input_batch: GeneratorInput, disable_tqdm: bool = False) -> GeneratorOutput:
        prompts = input_batch["prompts"]
        trajectory_ids = input_batch["trajectory_ids"]

        if trajectory_ids is None:
            raise ValueError("`trajectory_ids` is required in the input batch")
        if len(prompts) != len(trajectory_ids):
            raise ValueError(
                f"Prompt count ({len(prompts)}) doesn't match trajectory_ids count ({len(trajectory_ids)})"
            )

        # Captured once so every trajectory shares the policy version at the start of the batch.
        cache_salt = self._compute_cache_salt()

        all_outputs: List[HarborTrajectoryOutput] = [None] * len(prompts)  # type: ignore[list-item]
        all_attempts = [TrialAttempts() for _ in prompts]
        progress = tqdm(
            disable=disable_tqdm,  # disable for fully async training
            total=len(prompts),
            desc="Generating Trajectories",
            miniters=max(1, len(prompts) // 10),
            mininterval=5,
        )

        async def _worker(idx, prompt, trajectory_id):
            result = await self._harbor_agent_loop(
                prompt=prompt, trajectory_id=trajectory_id, cache_salt=cache_salt, attempts=all_attempts[idx]
            )
            all_outputs[idx] = result
            progress.update(1)

        try:
            async with asyncio.TaskGroup() as tg:
                for idx, (prompt, trajectory_id) in enumerate(zip(prompts, trajectory_ids)):
                    tg.create_task(_worker(idx, prompt, trajectory_id))
        finally:
            progress.close()

        output = build_step_wise_generator_output(
            all_outputs, overlong_filtering=self.generator_cfg.apply_overlong_filtering
        )
        output["rollout_metrics"].update(trial_metrics(all_attempts))
        return output

    async def _harbor_agent_loop(
        self,
        prompt: ConversationType,
        trajectory_id: TrajectoryID,
        cache_salt: Optional[str] = None,
        attempts: Optional[TrialAttempts] = None,
    ) -> HarborTrajectoryOutput:
        """Run a single Harbor trial and return the rollout details plus a trajectory-level reward.
        Remote trial failures are masked; transport and result-contract failures propagate.
        """
        agent_loop_start_time = time.monotonic()
        attempts = attempts if attempts is not None else TrialAttempts()
        reward = None
        results = None
        rollout_details = None
        num_turns = None
        successful = False
        is_context_length_error = False
        is_agent_timeout_error = False

        for i in range(MAX_NUM_RETRIES_PER_TRIAL):
            prefix = f"Trajectory {trajectory_id} attempt {i+1}/{MAX_NUM_RETRIES_PER_TRIAL}"
            results = None
            # Each attempt is a distinct router session; track it so it can be
            # released on completion/error/cancellation.
            session_id = uuid4().hex
            attempts.start()
            try:
                # Create a fresh Trial each attempt so agent state is clean on retry.
                config = deepcopy(self._harbor_trial_config_template)
                config["task"] = {"path": map_task_path(prompt, self.remote) if self.remote.endpoint else prompt}
                config["agent"]["kwargs"]["session_id"] = session_id
                # Forward the salt via llm_kwargs.extra_body -> LiteLLM -> the vLLM request's top-level
                # `cache_salt` field. vLLM rejects an empty salt, so attach only when set.
                if cache_salt is not None:
                    llm_kwargs = config["agent"]["kwargs"].setdefault("llm_kwargs", {})
                    extra_body = llm_kwargs.setdefault("extra_body", {})
                    if not isinstance(extra_body, dict):
                        raise TypeError("harbor_trial_config.agent.kwargs.llm_kwargs.extra_body must be a mapping")
                    extra_body["cache_salt"] = cache_salt
                async with self._rate_limiter:
                    if self.remote.endpoint:
                        # The CPU host validates with its own StateFork TrialConfig;
                        # the older Harbor on the GPU host must not strip search fields.
                        results = await run_remote_trial(config, self.remote)
                    else:
                        if config.get("search", {}).get("enabled") and "search" not in TrialConfig.model_fields:
                            raise RolloutContractError(
                                "Local Harbor has no search support; use the StateFork remote server"
                            )
                        trial_config = TrialConfig.model_validate(config)
                        trial = await Trial.create(trial_config)
                        results = await trial.run()
                attempts.record(results)

                exc_type = results.exception_info.exception_type if results.exception_info else None
                if (
                    self.remote.endpoint
                    and exc_type is not None
                    and not (exc_type == "TmuxSessionEndedError" and results.verifier_result is not None)
                ):
                    # Retain existing group masking for ordinary timeouts/startup errors.
                    # Every other reported trial failure masks only this trajectory.
                    stop_reason = {
                        "AgentTimeoutError": "agent_timeout",
                        "ContextLengthExceededError": "context_length",
                        "EnvironmentStartTimeoutError": "error",
                        "SandboxBuildFailedError": "error",
                        "WaypointShellDeadError": "error",
                    }.get(exc_type, "environment_error")
                    logger.warning(
                        f"{prefix}: {exc_type}: {results.exception_info.exception_message}; "
                        f"stop_reason={stop_reason} (no retry)."
                    )
                    return HarborTrajectoryOutput(
                        trajectory_id=trajectory_id,
                        stop_reason=stop_reason,
                        e2e_time=time.monotonic() - agent_loop_start_time,
                        exception_info=results.exception_info.model_dump(mode="json"),
                    )
                if self.remote.endpoint and results.search_limits_exhausted and exc_type is None:
                    logger.warning(f"{prefix}: search limits exhausted; masking this trajectory (no retry).")
                    return HarborTrajectoryOutput(
                        trajectory_id=trajectory_id,
                        stop_reason="search_limits_exhausted",
                        e2e_time=time.monotonic() - agent_loop_start_time,
                    )
                # Parse exception type
                is_context_length_error = exc_type == "ContextLengthExceededError"
                is_agent_timeout_error = exc_type == "AgentTimeoutError"

                # Determine reward.
                if is_agent_timeout_error:
                    # AgentTimeoutError: not successful, no retry, loss-masked
                    logger.debug(f"{prefix} hit AgentTimeoutError (no retry). Results: {results}")
                    break
                elif is_context_length_error:
                    # ContextLengthExceededError: always train with reward=0.
                    logger.debug(f"{prefix} hit ContextLengthExceededError, setting reward=0. Results: {results}")
                    reward = 0.0
                elif not results.verifier_result:
                    # Does not have a verifier result, so it's not successful, will retry
                    logger.warning(f"{prefix} failed: Exception info: {results.exception_info}. Results: {results}")
                    continue
                else:
                    reward = float(results.verifier_result.rewards["reward"])

                # Extract rollout details and check for success
                rollout_details, num_turns = extract_rollout(results.agent_result, reward)

                if (
                    rollout_details
                    and len(rollout_details) >= 1
                    and len(rollout_details[0].get("completion_token_ids", [])) > 0
                ):
                    successful = True
                    logger.debug(f"{prefix} successful: reward={reward}.")
                    break
                else:
                    if self.remote.endpoint:
                        raise ValueError("Remote Harbor returned no rollout tokens")
                    logger.warning(f"{prefix} failed: empty/missing rollout_details. Results: {results}")
            except RolloutContractError:
                raise
            except Exception as e:
                if self.remote.endpoint:
                    raise
                logger.warning(f"{prefix} failed: Error running trial: {e}. Results: {results}")
                attempts.fail(e)
                continue
            finally:
                await self.inference_engine_client.finish_session(session_id)

        if not successful:
            stop_reason = "agent_timeout" if is_agent_timeout_error else "error"
            error_message = f"Trajectory {trajectory_id} failed (stop_reason={stop_reason}), will set loss mask to [0]."
            if stop_reason == "error":
                error_message += f" Results: {results}"
            logger.warning(error_message)
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=None,
                stop_reason=stop_reason,
                e2e_time=time.monotonic() - agent_loop_start_time,
            )
        else:
            return HarborTrajectoryOutput(
                trajectory_id=trajectory_id,
                rollout_details=rollout_details,
                reward=reward,
                num_turns=num_turns,
                stop_reason="context_length" if is_context_length_error else "complete",
                e2e_time=time.monotonic() - agent_loop_start_time,
            )
