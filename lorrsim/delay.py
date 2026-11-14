"""Runtime delay generation, mirroring Start-Kit ``DelayGenerator``.

The official generator offers two *event* models and two *duration* models:

    eventModel    bernoulli : every available agent independently sampled
                             with probability ``pDelay`` each tick
                  poisson   : number of delayed agents per tick drawn from
                             Poisson(teamSize * pDelay)
    durationModel uniform   : U{minDelay, ..., maxDelay}
                  gaussian  : round(N(mean, std)) clamped to [minDelay, maxDelay]
                             with mean/std derived from the ratio parameters

Only the *family names* of the distributions are exposed to participant code in
the real Start-Kit (``env->delay_event_distribution`` /
``env->delay_time_distribution``); the numeric parameters are hidden.  Our
schedulers therefore never read this config -- they estimate delay risk online
from observed state, exactly as a real submission must.
"""

from __future__ import annotations

import random
from typing import List, Sequence

from .instance import DelayConfig


class DelayGenerator:
    def __init__(self, cfg: DelayConfig, num_agents: int):
        self.cfg = cfg
        self.num_agents = int(num_agents)
        self._rng = random.Random(cfg.seed)
        # Remaining blocked ticks per agent; >0 means "delayed this tick"
        self.remaining: List[int] = [0] * self.num_agents
        self.intervals: List[List[List[int]]] = [[] for _ in range(self.num_agents)]
        self._open_start: List[int] = [-1] * self.num_agents

    # ---- duration sampling ---------------------------------------------
    def _sample_duration(self) -> int:
        c = self.cfg
        lo, hi = int(c.min_delay), int(c.max_delay)
        if hi < lo:
            lo, hi = hi, lo
        if c.duration_model == "gaussian":
            span = max(1, hi - lo)
            mean = lo + c.gauss_mean_ratio * span
            std = max(1e-6, c.gauss_std_ratio * span)
            v = int(round(self._rng.gauss(mean, std)))
            return max(lo, min(hi, v))
        return self._rng.randint(lo, hi)

    # ---- per-tick event generation -------------------------------------
    def tick(self, available: Sequence[bool], timestep: int) -> List[bool]:
        """Advance one tick and return a per-agent ``delayed`` mask.

        ``available[i]`` is False for robots that already cannot move this tick
        for other reasons; the real generator only delays agents that could
        otherwise have acted.
        """
        n = self.num_agents
        delayed = [False] * n

        # 1) continue already-running delays
        for i in range(n):
            if self.remaining[i] > 0:
                self.remaining[i] -= 1
                delayed[i] = True
                if self.remaining[i] == 0:
                    start = self._open_start[i]
                    if start >= 0:
                        self.intervals[i].append([start, timestep])
                        self._open_start[i] = -1

        # 2) sample new delay events for eligible robots
        c = self.cfg
        if c.p_delay > 0.0:
            eligible = [
                i for i in range(n) if available[i] and not delayed[i]
            ]
            if eligible:
                if c.event_model == "poisson":
                    lam = c.p_delay * max(1, len(eligible))
                    k = self._poisson(lam)
                else:
                    k = sum(1 for i in eligible if self._rng.random() < c.p_delay)
                k = min(k, len(eligible))
                if k > 0:
                    chosen = self._rng.sample(eligible, k)
                    for i in chosen:
                        dur = self._sample_duration()
                        if dur <= 0:
                            continue
                        delayed[i] = True
                        if self._open_start[i] < 0:
                            self._open_start[i] = timestep
                        self.remaining[i] = dur - 1
                        if self.remaining[i] == 0:
                            self.intervals[i].append([timestep, timestep + 1])
                            self._open_start[i] = -1
        return delayed

    def _poisson(self, lam: float) -> int:
        """Knuth's algorithm; lam is small in practice (< a few)."""
        if lam <= 0:
            return 0
        if lam > 30:  # normal approximation keeps it O(1) for huge fleets
            v = int(round(self._rng.gauss(lam, lam ** 0.5)))
            return max(0, v)
        import math

        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            p *= self._rng.random()
            if p <= L:
                return k
            k += 1

    # ---- reporting -----------------------------------------------------
    def flush(self, timestep: int) -> None:
        for i in range(self.num_agents):
            if self._open_start[i] >= 0:
                self.intervals[i].append([self._open_start[i], timestep])
                self._open_start[i] = -1

    def total_delayed_ticks(self) -> int:
        return sum(max(0, b - a) for ivs in self.intervals for a, b in ivs)
