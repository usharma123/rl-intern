from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rl_intern.orchestrator.models import ExperimentPlan


class DomainAdapter(ABC):
    domain: str

    @abstractmethod
    def inspect(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def prepare(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def smoke_test(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def train(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def report(self, plan: ExperimentPlan, run_dir: str | None = None) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def artifact_schema(self) -> dict[str, Any]:
        raise NotImplementedError


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, DomainAdapter] = {}

    def register(self, adapter: DomainAdapter) -> None:
        self._adapters[adapter.domain] = adapter

    def get(self, domain: str) -> DomainAdapter:
        try:
            return self._adapters[domain]
        except KeyError as exc:
            raise ValueError(f"Unknown RL domain adapter: {domain}") from exc

    def list_domains(self) -> list[str]:
        return sorted(self._adapters)


_REGISTRY: AdapterRegistry | None = None


def get_registry() -> AdapterRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        from rl_intern.domains.gym_sb3.adapter import GymSB3Adapter
        from rl_intern.domains.llm_trl.adapter import LLMTRLAdapter

        registry = AdapterRegistry()
        registry.register(GymSB3Adapter())
        registry.register(LLMTRLAdapter())
        _REGISTRY = registry
    return _REGISTRY


def get_adapter(domain: str) -> DomainAdapter:
    return get_registry().get(domain)
