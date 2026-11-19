"""ForkRL group transport using SkyRL's native reward-based training path."""

import asyncio
import math
import time
from collections import defaultdict
from uuid import uuid4

import httpx
from loguru import logger

from .harbor_generator import (
    HarborGenerator,
    HarborTrajectoryOutput,
    build_step_wise_generator_output,
)
from .remote import map_task_path, post_idempotent_json


def group_trajectories(result, trajectory_ids, elapsed, *, use_step_advantages=False):
    """Translate a group into turns, optionally retaining service-computed advantages."""
    if result["status"] == "failed":
        logger.warning("ForkRL group failed: {}", result.get("error"))
        return [
            HarborTrajectoryOutput(
                tid,
                stop_reason="error",
                e2e_time=elapsed,
                step_advantages=[] if use_step_advantages else None,
            )
            for tid in trajectory_ids
        ]
    if result.get("schema") != "forkrl.group_result/v0":
        raise ValueError("ForkRL returned no group result")
    if not use_step_advantages and (result["start"]["kind"] != "s0" or result["credit_config"]["rule"] != "off"):
        raise ValueError("Native GRPO/DAPO requires ForkRL start=s0 and credit.rule=off")
    rollouts = result["rollouts"]
    indexed = ["index" in rollout for rollout in rollouts]
    if all(indexed):
        by_id = {rollout["index"]: rollout for rollout in rollouts}
    elif any(indexed):
        raise ValueError("ForkRL mixed indexed and unindexed rollouts")
    else:
        # The current API preserves group order and identifies each rollout by an opaque ID.
        ids = [rollout.get("trajectory_id") for rollout in rollouts]
        if any(not isinstance(tid, str) or not tid for tid in ids) or len(set(ids)) != len(ids):
            raise ValueError("ForkRL returned missing or duplicate trajectory IDs")
        by_id = dict(enumerate(rollouts))
    expected = list(range(len(trajectory_ids)))
    if len(rollouts) != len(expected) or set(by_id) != set(expected):
        raise ValueError("ForkRL returned the wrong rollout IDs or group size")
    outputs = []
    for key, tid in zip(expected, trajectory_ids, strict=True):
        rollout = by_id[key]
        stop = rollout["stop_reason"]
        if stop not in {"complete", "agent_timeout", "context_limit", "infra_error"}:
            raise ValueError(f"Unknown ForkRL stop reason: {stop}")
        if result["masked"] or stop == "infra_error":
            outputs.append(
                HarborTrajectoryOutput(
                    tid,
                    stop_reason="error" if result["masked"] else "environment_error",
                    e2e_time=elapsed,
                    exception_info={"exception_message": rollout.get("error")},
                    step_advantages=[] if use_step_advantages else None,
                )
            )
            continue
        turns = rollout["turns"]
        reward = float(rollout["reward"])
        if not math.isfinite(reward):
            raise ValueError("ForkRL returned a non-finite reward")
        if not turns and stop != "context_limit":
            raise ValueError("Scored ForkRL rollout has no turns")
        details = {
            field: [turn[field] for turn in turns] for field in ("prompt_token_ids", "completion_token_ids", "logprobs")
        }
        step_advantages = None
        if use_step_advantages:
            turn_ids = [turn["turn"] for turn in turns]
            if any(not isinstance(t, int) or t < 1 for t in turn_ids) or any(
                left >= right for left, right in zip(turn_ids, turn_ids[1:])
            ):
                raise ValueError("ForkRL turns must have increasing positive turn IDs")
            step_advantages = [turn.get("advantage") for turn in turns]
            if any(not isinstance(a, (int, float)) or not math.isfinite(a) for a in step_advantages):
                raise ValueError("Every trainable ForkRL turn must have a finite advantage")
        for turn in turns:
            if (
                not turn["prompt_token_ids"]
                or not turn["completion_token_ids"]
                or len(turn["completion_token_ids"]) != len(turn["logprobs"])
            ):
                raise ValueError("ForkRL turn has missing or misaligned tokens/logprobs")
        # An arm stopped at its budget can still have a valid verifier reward.
        outputs.append(
            HarborTrajectoryOutput(
                tid,
                rollout_details=[details] if turns else None,
                reward=reward,
                num_turns=len(turns),
                stop_reason="context_length" if stop == "context_limit" else "complete",
                e2e_time=elapsed,
                step_advantages=step_advantages,
            )
        )
    return outputs


class ForkRLGenerator(HarborGenerator):
    """Submit once per task group, poll, and reuse Harbor's step-wise output conversion."""

    def __init__(self, *args, run_id: str = "default", **kwargs):
        super().__init__(*args, **kwargs)
        if not self.remote.endpoint:
            raise ValueError("ForkRL requires generator.remote.endpoint")
        self.run_id = run_id
        # Filled by HarborExp after the trainer computes its training horizon.
        self.total_steps: int | None = None

    async def generate(self, input_batch, disable_tqdm=False):
        prompts = input_batch["prompts"]
        tids = input_batch["trajectory_ids"]
        if tids is None or len(tids) != len(prompts):
            raise ValueError("ForkRL requires one trajectory ID per prompt")
        groups = defaultdict(list)
        for index, tid in enumerate(tids):
            groups[tid.instance_id].append(index)
        metadata = input_batch.get("batch_metadata")
        step = metadata.global_step if metadata is not None else 0
        version = getattr(self.inference_engine_client, "weight_version", None)
        version = step if version is None else version
        outputs = [None] * len(prompts)

        async def run(indices):
            paths = {map_task_path(prompts[i], self.remote) for i in indices}
            if len(paths) != 1:
                raise ValueError("A ForkRL group must contain exactly one task")
            path = paths.pop()
            # A short task label also bounds Waypoint's filesystem/socket path length.
            body = {
                "schema": "forkrl.group_request/v0",
                "request_id": uuid4().hex,
                "run_id": self.run_id,
                "step": step,
                "policy_version": version,
                "task": {"id": path.rstrip("/").rsplit("/", 1)[-1], "path": path},
                "n": len(indices),
                "deadline_sec": self.remote.timeout_sec,
            }
            if self.total_steps is not None:
                body["total_steps"] = self.total_steps
            if self.remote.max_turns is not None:
                body["max_turns"] = self.remote.max_turns
            try:
                result, elapsed = await self._run_remote_group(body)
            except TimeoutError:
                logger.error(
                    "ForkRL request {} exceeded its {}s total deadline; masking this group",
                    body["request_id"],
                    self.remote.timeout_sec,
                )
                for index in indices:
                    outputs[index] = HarborTrajectoryOutput(
                        tids[index],
                        stop_reason="error",
                        e2e_time=self.remote.timeout_sec,
                        exception_info={"exception_message": "Remote group deadline exceeded"},
                        step_advantages=[] if self.remote.use_step_advantages else None,
                    )
                return
            try:
                converted = group_trajectories(
                    result,
                    [tids[i] for i in indices],
                    elapsed,
                    use_step_advantages=self.remote.use_step_advantages,
                )
                for index, output in zip(indices, converted, strict=True):
                    outputs[index] = output
                # Receipt bookkeeping must not discard an already validated result.
                try:
                    async with httpx.AsyncClient(timeout=httpx.Timeout(65, connect=10), trust_env=False) as client:
                        response = await client.post(
                            f"{self.remote.endpoint.rstrip('/')}/v0/ack", json={"group_id": result["group_id"]}
                        )
                        response.raise_for_status()
                except httpx.HTTPError as exc:
                    logger.warning(
                        "ForkRL acknowledgement failed for {}; keeping received training data: {!r}",
                        result["group_id"],
                        exc,
                    )
            finally:
                await asyncio.gather(
                    *(
                        self.inference_engine_client.finish_session(r["session_id"])
                        for r in result.get("rollouts", [])
                        if r.get("session_id")
                    )
                )

        async with asyncio.TaskGroup() as tasks:
            for indices in groups.values():
                tasks.create_task(run(indices))
        return build_step_wise_generator_output(outputs, overlong_filtering=self.generator_cfg.apply_overlong_filtering)

    async def _run_remote_group(self, body):
        started = time.monotonic()
        endpoint = self.remote.endpoint.rstrip("/")
        # The CPU service owns sandbox capacity; each submission already contains all N arms.
        async with asyncio.timeout(self.remote.timeout_sec):
            async with httpx.AsyncClient(timeout=httpx.Timeout(65, connect=10), trust_env=False) as client:
                status = await post_idempotent_json(client, f"{endpoint}/v0/submit", body)
                group_id = status["group_id"]
                while True:
                    result = await post_idempotent_json(
                        client, f"{endpoint}/v0/poll", {"group_id": group_id, "wait_sec": 30}
                    )
                    if result.get("request_id") != body["request_id"] or result.get("group_id") != group_id:
                        raise ValueError("ForkRL returned a result for another request")
                    if result["status"] in {"done", "failed", "acked"}:
                        return result, time.monotonic() - started
