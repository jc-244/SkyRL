"""Run vanilla Harbor through Tinker Cookbook's official async training loop."""

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass

import chz
import tinker
from datasets import load_dataset
from tinker_cookbook import model_info
from tinker_cookbook.completers import TinkerTokenCompleter, TokensWithLogprobs
from tinker_cookbook.exceptions import AllTrajectoriesFailedError
from tinker_cookbook.rl import rollouts, train
from tinker_cookbook.rl.types import (
    EnvGroupBuilder,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
    TrajectoryGroup,
    Transition,
)

from examples.train_integrations.harbor.tinker_runner import HarborRunner, TinkerTerminus2

logger = logging.getLogger(__name__)
_policy: ContextVar[TinkerTokenCompleter] = ContextVar("harbor_tinker_policy")


class CookbookTerminus2(TinkerTerminus2):
    """Pin every turn of a trial to the sampler supplied by Cookbook."""

    def _init_llm(self, model_name, temperature, collect_rollout_details, llm_kwargs, **kwargs):
        policy = _policy.get()
        options = dict(llm_kwargs or {})
        options["max_tokens"] = policy.max_tokens
        llm = super()._init_llm(model_name, policy.temperature, collect_rollout_details, options, **kwargs)
        llm._sampling_client = policy.sampling_client
        return llm


@dataclass
class HarborGroup(EnvGroupBuilder):
    row: dict
    runner: HarborRunner

    async def make_envs(self):
        raise NotImplementedError("Harbor creates environments inside its Trial; use the Harbor rollout hook")

    def logging_tags(self):
        return [self.row.get("data_source", "harbor")]


def as_trajectory_group(output):
    """Preserve original tokens/logprobs, group membership, and masked members."""
    if not any(output["episode_nums"]):
        raise AllTrajectoriesFailedError("Harbor group contains no valid training turns")
    turns = iter(zip(output["prompt_token_ids"], output["response_ids"], output["rollout_logprobs"], strict=True))
    trajectories, metrics = [], []
    for count in output["episode_nums"]:
        transitions = []
        for index in range(count):
            prompt, response, logprobs = next(turns)
            transitions.append(
                Transition(
                    ob=tinker.ModelInput.from_ints(prompt),
                    ac=TokensWithLogprobs(response, logprobs),
                    reward=0.0,
                    episode_done=index == count - 1,
                )
            )
        trajectories.append(Trajectory(transitions, final_ob=tinker.ModelInput.from_ints([])))
        metrics.append({"harbor/masked": int(count == 0)})
    return TrajectoryGroup(trajectories, output["traj_rewards"], metrics)


async def harbor_group_rollout(env_group_builder, policy, strategy=None, termination=None):
    """Cookbook's documented do_group_rollout replacement; Harbor owns the agent loop."""
    if not isinstance(env_group_builder, HarborGroup) or not isinstance(policy, TinkerTokenCompleter):
        raise TypeError("The Harbor rollout hook requires HarborGroup and a Tinker sampling policy")
    token = _policy.set(policy)
    try:
        output = await env_group_builder.runner.run([env_group_builder.row])
        logger.info("Harbor group %s: %s", env_group_builder.row["task_path"], output["rollout_metrics"])
        return as_trajectory_group(output)
    finally:
        _policy.reset(token)


class HarborDataset(RLDataset):
    def __init__(self, rows, runner, groups_per_batch):
        self.rows = rows
        self.runner = runner
        self.groups_per_batch = groups_per_batch

    def __len__(self):
        return len(self.rows) // self.groups_per_batch

    def get_batch(self, index):
        start = index * self.groups_per_batch
        return [HarborGroup(self.rows[i], self.runner) for i in range(start, start + self.groups_per_batch)]


@chz.chz
class HarborDatasetBuilder(RLDatasetBuilder):
    dataset_file: str
    harbor_trial_config: str
    model_name: str
    groups_per_batch: int
    group_size: int
    harbor_max_concurrency: int
    renderer_name: str
    seed: int = 0

    async def __call__(self):
        rows = load_dataset("parquet", data_files=self.dataset_file, split="train").shuffle(seed=self.seed)
        if "task_path" not in rows.column_names or len(rows) < self.groups_per_batch:
            raise ValueError("Training data needs task_path and at least groups_per_batch rows")
        runner = HarborRunner(
            self.harbor_trial_config, self.model_name, None, self.group_size, self.harbor_max_concurrency
        )
        runner.config["agent"]["import_path"] = f"{CookbookTerminus2.__module__}:CookbookTerminus2"
        runner.config["agent"]["kwargs"]["llm_kwargs"]["renderer_name"] = self.renderer_name
        return HarborDataset(rows, runner, self.groups_per_batch), None


@chz.chz
class Config:
    dataset_file: str
    harbor_trial_config: str
    log_path: str
    model_name: str = "Qwen/Qwen3.5-9B"
    groups_per_batch: int = 1
    group_size: int = 8
    harbor_max_concurrency: int = 8
    max_steps_off_policy: int = 4
    max_steps: int = 1
    num_epochs: int = 1
    learning_rate: float = 4e-5
    lora_rank: int = 16
    max_tokens: int = 8192
    temperature: float = 1.0
    save_every: int = 2
    seed: int = 0
    wandb_project: str | None = None
    wandb_name: str | None = None
    base_url: str | None = None


async def main(config: Config):
    if config.groups_per_batch < 1 or config.max_steps_off_policy < 0:
        raise ValueError("groups_per_batch must be positive and max_steps_off_policy nonnegative")
    renderer_name = model_info.get_recommended_renderer_name(config.model_name)
    dataset_builder = HarborDatasetBuilder(
        dataset_file=config.dataset_file,
        harbor_trial_config=config.harbor_trial_config,
        model_name=config.model_name,
        groups_per_batch=config.groups_per_batch,
        group_size=config.group_size,
        harbor_max_concurrency=config.harbor_max_concurrency,
        renderer_name=renderer_name,
        seed=config.seed,
    )
    official_config = train.Config(
        dataset_builder=dataset_builder,
        model_name=config.model_name,
        renderer_name=renderer_name,
        recipe_name="skyrl_harbor_async",
        learning_rate=config.learning_rate,
        lora_rank=config.lora_rank,
        loss_fn="ppo",
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        async_config=train.AsyncConfig(
            max_steps_off_policy=config.max_steps_off_policy, groups_per_batch=config.groups_per_batch
        ),
        max_steps=config.max_steps,
        num_epochs=config.num_epochs,
        save_every=config.save_every,
        eval_every=0,
        log_path=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        base_url=config.base_url,
        enable_trace=True,
    )
    # This hook is documented in Cookbook's rl/train.py. Scope it to this process/run.
    original = rollouts.do_group_rollout
    rollouts.do_group_rollout = harbor_group_rollout
    try:
        await train.main(official_config)
    finally:
        rollouts.do_group_rollout = original


if __name__ == "__main__":
    asyncio.run(main(chz.entrypoint(Config)))
