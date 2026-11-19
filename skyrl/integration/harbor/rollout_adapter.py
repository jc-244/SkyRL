"""Select the search submission whose verifier reward SkyRL will train on."""

from typing import Any


class RolloutContractError(ValueError):
    """Search results cannot be mapped to one rewarded conversation."""


def extract_rollout(agent_result: Any, reward: float) -> tuple[list[dict], int]:
    details = agent_result.rollout_details if agent_result is not None else None
    search = (agent_result.metadata or {}).get("search") if agent_result is not None else None
    if search is not None:
        selected = search["selected_node_id"]
        verdict = search["selected_verification"]
        if selected not in verdict["node_ids"] or verdict["error"] is not None or verdict["reward"] != reward:
            raise RolloutContractError("Search submission and verifier reward do not match")
        details = [
            detail for node_id, detail in zip(search["rollout_node_ids"], details, strict=True) if node_id == selected
        ]
        if len(details) != 1:
            raise RolloutContractError("Search submission must contain one conversation segment")
    # SkyRL checks token/logprob shapes when building its step-wise output.
    return (details, len(details[0]["completion_token_ids"])) if details else ([], 0)
