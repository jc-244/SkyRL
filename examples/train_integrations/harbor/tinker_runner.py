"""Adapt vanilla Harbor trials to the existing SkyRL Agent Tinker trainer."""

import asyncio
import logging
import math
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

import yaml
from harbor.agents.terminus_2 import Terminus2
from harbor.llms.base import OutputLengthExceededError
from harbor.llms.tinker import TinkerLLM
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial
from tinker_cookbook.renderers import get_text_content

logger = logging.getLogger(__name__)
LENGTH_ERRORS = {"ContextLengthExceededError", "OutputLengthExceededError"}


class TextTinkerLLM(TinkerLLM):
    """Bridge Cookbook content blocks to Harbor 0.15's text-only command parser."""

    async def call(self, *args, **kwargs):
        try:
            response = await super().call(*args, **kwargs)
        except OutputLengthExceededError as error:
            if error.truncated_response is not None:
                error.truncated_response = get_text_content({"content": error.truncated_response})
            raise
        response.content = get_text_content({"content": response.content})
        return response


class TinkerTerminus2(Terminus2):
    """Keep the standard agent loop, using the text-compatible Tinker backend."""

    def _init_llm(self, model_name, temperature, collect_rollout_details, llm_kwargs, **kwargs):
        options = dict(llm_kwargs or {})
        if temperature is not None:
            options["temperature"] = temperature
        return TextTinkerLLM(model_name=model_name, collect_rollout_details=collect_rollout_details, **options)


class HarborRunner:
    """Run independent trials; keep each question's samples adjacent for GRPO."""

    def __init__(self, config_path, model_name, model_path, group_size, max_concurrency):
        if group_size is None or group_size < 1 or max_concurrency < 1:
            raise ValueError("Harbor requires positive group_size and max_concurrency")
        self.config = yaml.safe_load(Path(config_path).read_text())
        agent = self.config.setdefault("agent", {})
        if agent.get("name") != "terminus-2":
            raise ValueError("The Harbor Tinker adapter requires terminus-2")
        agent.update(name=None, import_path=f"{__name__}:TinkerTerminus2")
        kwargs = agent.setdefault("kwargs", {})
        if kwargs.get("enable_summarize", False):
            raise ValueError("Disable summarization to retain complete training token records")
        agent["model_name"] = model_name
        kwargs.update(llm_backend="tinker", collect_rollout_details=True, enable_summarize=False)
        kwargs.setdefault("llm_kwargs", {})["model_path"] = model_path
        self.context_limit = kwargs["llm_kwargs"].get("context_limit", 32000)
        self.group_size = group_size
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _trial(self, row):
        config = deepcopy(self.config)
        config["task"] = {"path": str(Path(row["task_path"]).expanduser())}
        config["trial_name"] = f"tinker-{uuid4().hex}"
        async with self.semaphore:
            trial = await Trial.create(TrialConfig.model_validate(config))
            return await trial.run()

    def _extract(self, result):
        """Return reward, per-turn tokens, and whether an error invalidates the group."""
        error = result.exception_info
        if error is not None:
            logger.warning("Trial %s: %s", result.trial_name, error.exception_type)
            return 0.0, [], error.exception_type not in LENGTH_ERRORS
        if result.agent_result is None or result.verifier_result is None:
            logger.warning("Trial %s has no agent or verifier result", result.trial_name)
            return 0.0, [], True
        details = result.agent_result.rollout_details
        rewards = result.verifier_result.rewards
        if not details or not rewards or "reward" not in rewards:
            logger.warning("Trial %s has no training tokens or reward", result.trial_name)
            return 0.0, [], True
        # This adapter collects linear conversations, not StateFork branches.
        if len(details) != 1:
            raise ValueError("Expected one linear conversation per Harbor trial")
        detail = details[0]
        turns = list(zip(detail["prompt_token_ids"], detail["completion_token_ids"], detail["logprobs"], strict=True))
        reward = float(rewards["reward"])
        if not math.isfinite(reward) or not turns:
            raise ValueError(f"Invalid reward or empty conversation: {result.trial_name}")
        for prompt, response, logprobs in turns:
            if not prompt or not response or len(response) != len(logprobs):
                raise ValueError(f"Missing or misaligned training tokens: {result.trial_name}")
            if not all(math.isfinite(value) for value in logprobs):
                raise ValueError(f"Non-finite logprobs: {result.trial_name}")
            if len(prompt) + len(response) > self.context_limit:
                logger.warning("Trial %s exceeds the training context limit", result.trial_name)
                return 0.0, [], False
        return reward, turns, False

    async def run(self, input_batch, val_mode=False):
        group_size = 1 if val_mode else self.group_size
        # TaskGroup cancels outstanding work if configuration/programming errors escape Trial.run.
        async with asyncio.TaskGroup() as tasks:
            pending = [tasks.create_task(self._trial(row)) for row in input_batch for _ in range(group_size)]
        output = {
            key: []
            for key in (
                "prompt_token_ids",
                "response_ids",
                "rollout_logprobs",
                "loss_masks",
                "traj_rewards",
                "episode_nums",
            )
        }
        failed_groups = 0
        masked = 0
        for start in range(0, len(pending), group_size):
            group = [self._extract(task.result()) for task in pending[start : start + group_size]]
            failed_group = any(failed for _, _, failed in group)
            failed_groups += int(failed_group)
            for reward, turns, _ in group:
                if failed_group:
                    reward, turns = 0.0, []
                masked += int(not turns)
                output["traj_rewards"].append(reward)
                output["episode_nums"].append(len(turns))
                for prompt, response, logprobs in turns:
                    output["prompt_token_ids"].append(prompt)
                    output["response_ids"].append(response)
                    output["rollout_logprobs"].append(logprobs)
                    output["loss_masks"].append([1] * len(response))
        output["rollout_metrics"] = {
            "trials": len(pending),
            "masked_trials": masked,
            "failed_groups": failed_groups,
            "training_turns": len(output["response_ids"]),
        }
        return output
