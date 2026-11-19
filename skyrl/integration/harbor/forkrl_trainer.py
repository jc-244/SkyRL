"""Use ForkRL's per-turn credit with SkyRL's existing synchronous and async trainers."""

import torch

from skyrl.train.fully_async_trainer import FullyAsyncRayPPOTrainer
from skyrl.train.trainer import RayPPOTrainer


def validate_forkrl_credit_config(cfg):
    """Reject settings that would merge, discard or recompute remote per-turn credit."""
    if not cfg.generator.remote.use_step_advantages:
        return
    required = {
        "generator.remote.protocol": (cfg.generator.remote.protocol, "forkrl"),
        "generator.step_wise_trajectories": (cfg.generator.step_wise_trajectories, True),
        "generator.merge_stepwise_output": (cfg.generator.merge_stepwise_output, False),
        "trainer.algorithm.zero_variance_filter": (cfg.trainer.algorithm.zero_variance_filter, False),
        "trainer.fully_async.sample_full_batch": (cfg.trainer.fully_async.sample_full_batch, False),
        "trainer.algorithm.advantage_batch_normalize": (cfg.trainer.algorithm.advantage_batch_normalize, False),
        "trainer.algorithm.use_kl_in_reward": (cfg.trainer.algorithm.use_kl_in_reward, False),
        "trainer.critic.model.path": (cfg.trainer.critic.model.path, None),
    }
    conflicts = [f"{name}={expected}" for name, (actual, expected) in required.items() if actual != expected]
    if conflicts:
        raise ValueError("ForkRL per-turn advantages require: " + ", ".join(conflicts))


class ForkRLAdvantagesMixin:
    """Keep per-turn advantages aligned through tensor conversion and DP padding."""

    def __init__(self, cfg, *args, **kwargs):
        validate_forkrl_credit_config(cfg)
        super().__init__(cfg, *args, **kwargs)

    def convert_to_training_input(self, generator_output, uids):
        advantages = generator_output.get("step_advantages")
        if advantages is None or len(advantages) != len(generator_output["response_ids"]):
            raise ValueError("ForkRL requires one advantage per training row")
        values = torch.tensor(advantages, dtype=torch.float32)
        if values.ndim != 1 or not torch.isfinite(values).all():
            raise ValueError("ForkRL per-turn advantages must be finite scalars")
        data = super().convert_to_training_input(generator_output, uids)
        data.metadata["step_advantages"] = list(advantages) + [0.0] * data.metadata["pad_size"]
        return data

    @torch.no_grad()
    def compute_advantages_and_returns(self, data):
        response_mask = data["response_mask"]
        values = torch.as_tensor(data.metadata.pop("step_advantages"), dtype=torch.float32, device=response_mask.device)
        if values.shape != (data.batch_size,) or not torch.isfinite(values).all():
            raise ValueError("ForkRL advantages do not match the padded training batch")
        mask = response_mask.bool() & data["loss_mask"].bool()
        advantages = values[:, None] * mask
        data["advantages"] = advantages
        data["returns"] = advantages  # No critic: these are not value-function targets.

        num_real = data.batch_size - data.metadata["pad_size"]
        last = torch.as_tensor(
            data.metadata["is_last_step"][:num_real], dtype=torch.bool, device=data["rewards"].device
        )
        rewards = data["rewards"][:num_real].sum(-1)[last]
        valid = advantages[:num_real][mask[:num_real]]
        metrics = {
            "avg_final_rewards": rewards.mean().item() if rewards.numel() else 0.0,
            "avg_response_length": data.metadata["avg_response_length"],
            "avg_advantages": valid.mean().item() if valid.numel() else 0.0,
            "avg_advantages_abs": valid.abs().mean().item() if valid.numel() else 0.0,
        }
        data.metadata.setdefault("metrics", {}).update(metrics)
        self.all_metrics.update(
            {
                "loss/avg_final_rewards": metrics["avg_final_rewards"],
                "loss/avg_raw_advantages": metrics["avg_advantages"],
                "loss/avg_raw_advantages_abs": metrics["avg_advantages_abs"],
            }
        )
        return data.to("cpu")


class ForkRLPPOTrainer(ForkRLAdvantagesMixin, RayPPOTrainer):
    """Synchronous SkyRL updates with externally assigned per-turn advantages."""


class ForkRLFullyAsyncTrainer(ForkRLAdvantagesMixin, FullyAsyncRayPPOTrainer):
    """Native fully-async SkyRL updates with externally assigned per-turn advantages."""
