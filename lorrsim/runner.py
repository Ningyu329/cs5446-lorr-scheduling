"""Simulation entry point -- the ``run_sim`` used by every script."""

from __future__ import annotations

from typing import Optional

from .instance import Instance, load_instance
from .schedulers import make_scheduler
from .world import Simulation, SimResult


def run_sim(
    instance: Instance,
    scheduler_key: str = "S0",
    *,
    seed: int = 0,
    horizon: int = 1200,
    plan_comm: int = 20,
    sched_budget_ms: float = 50.0,
    scheduler_kwargs: Optional[dict] = None,
    collect_trace: bool = False,
) -> SimResult:
    sched = make_scheduler(scheduler_key, **(scheduler_kwargs or {}))
    sim = Simulation(
        instance,
        sched,
        seed=seed,
        horizon=horizon,
        plan_comm=plan_comm,
        sched_budget_ms=sched_budget_ms,
        collect_trace=collect_trace,
    )
    return sim.run()


def run_from_path(path: str, scheduler_key: str = "S0", **kwargs) -> SimResult:
    return run_sim(load_instance(path), scheduler_key, **kwargs)
