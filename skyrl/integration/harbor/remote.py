"""Run a Harbor trial on a CPU host and await its result."""

import asyncio
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

import httpx
from harbor.models.agent.context import AgentContext
from harbor.models.trial.result import ExceptionInfo, TimingInfo
from harbor.models.verifier.result import VerifierResult
from loguru import logger
from pydantic import BaseModel


@dataclass
class RemoteHarborConfig:
    protocol: Literal["trial", "forkrl"] = "trial"
    """Remote API: individual Harbor trials or ForkRL rollout groups."""
    use_step_advantages: bool = False
    """Use ForkRL's per-turn advantages instead of SkyRL's reward-based estimator."""
    max_turns: int | None = None
    """ForkRL per-rollout turn limit; None uses the CPU service's configured limit."""
    endpoint: str | None = None
    """CPU-host service URL reached through SSH; None runs Harbor locally."""
    inference_url: str | None = None
    """SkyRL's /v1 inference URL as reached from the CPU host."""
    local_task_root: str | None = None
    """Dataset root on the GPU host; symlinks are resolved before mapping paths."""
    remote_task_root: str | None = None
    """Absolute path to the same dataset root on the CPU host."""
    timeout_sec: float = 7200.0
    """Maximum time for one request, including queueing and trial execution."""


class RemoteTrialResult(BaseModel):
    """Trial fields consumed by SkyRL, without the version-specific TrialConfig."""

    agent_result: AgentContext | None = None
    verifier_result: VerifierResult | None = None
    exception_info: ExceptionInfo | None = None
    environment_setup: TimingInfo | None = None
    agent_execution: TimingInfo | None = None
    verifier: TimingInfo | None = None

    @property
    def search_limits_exhausted(self) -> bool:
        search = (self.agent_result.metadata or {}).get("search", {}) if self.agent_result else {}
        return search.get("status") == "limits_exhausted"


def map_task_path(path: str, remote: RemoteHarborConfig) -> str:
    root = Path(remote.local_task_root).expanduser().resolve()
    relative = Path(path).expanduser().resolve().relative_to(root)
    target = PurePosixPath(remote.remote_task_root)
    if not target.is_absolute():
        raise ValueError("remote_task_root must be absolute")
    return str(target / relative)


async def post_idempotent_json(client: httpx.AsyncClient, url: str, body: dict) -> dict:
    """Retry the same idempotent operation within the caller's overall deadline."""
    delay = 1.0
    while True:
        try:
            response = await client.post(url, json=body)
            if response.status_code not in {408, 429, 502, 503, 504}:
                response.raise_for_status()
                return response.json()
            reason = f"HTTP {response.status_code}"
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            reason = repr(exc)
        logger.warning("Remote request {} failed ({}); retrying the same request/group in {}s", url, reason, delay)
        await asyncio.sleep(delay)
        delay = min(delay * 2, 10.0)


async def run_remote_trial(config: dict, remote: RemoteHarborConfig) -> RemoteTrialResult:
    # Closing this request on timeout/cancellation tells the CPU host to stop the trial.
    async with asyncio.timeout(remote.timeout_sec):
        async with httpx.AsyncClient(timeout=httpx.Timeout(remote.timeout_sec, connect=10), trust_env=False) as client:
            response = await client.post(f"{remote.endpoint.rstrip('/')}/trial", json=config)
            response.raise_for_status()
            result = RemoteTrialResult.model_validate(response.json())
    # A valid trial failure is data, regardless of its Python exception name.
    # HTTP, schema, and cancellation errors still propagate to the caller.
    if result.exception_info is not None or result.search_limits_exhausted:
        return result
    if result.agent_result is None or result.verifier_result is None:
        raise ValueError("Remote Harbor returned no agent or verifier result")
    return result
