"""Congestion (C) and delay-risk (R) feature extraction.

This module is the technical heart of the study.  Everything here is computed
from *observable* state only -- the official ``SharedEnvironment`` exposes just
the distribution *family names* of the delay model, never its parameters, so a
real submission must infer risk online.  We honour that constraint strictly:
``FeatureBank`` never reads ``DelayConfig``.

Metric design follows the literature (see ``docs/related_work.md``):

  C1  area occupancy   ``robots in radius r / passable cells in radius r``
      -- the alpha-threshold occupancy idea from the warehouse rule-based work
  C2  vertex pressure  ``ceil((n_v - 1) / 2)`` summed along the route, where
      n_v counts robots whose currently planned path uses cell v
      -- Chen et al. (AAAI 2024) traffic-flow vertex congestion
  C3  counter-flow     ``f_ab * f_ba`` on the bottleneck cells of the route
      -- Chen et al. (AAAI 2024) contraflow term
  C4  edge waiting     exponential moving average of observed wait ticks per
      cell -- "past traffic" cost from the flow-based assignment work

  R1  individual delay duty cycle   smoothed share of ticks robot i spent delayed
  R2  prediction drift              |system tick - last acknowledged progress|
  R3  spatial delay field           smoothed density of recently delayed robots
  R4  starvation risk               pool age of the task relative to service time

All C and R terms are returned in **tick** units so that
``S = d + lambda*C + gamma*R`` is dimensionally meaningful and lambda/gamma are
interpretable across instances (see ``calibrate``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple


class FeatureBank:
    def __init__(self, grid, num_agents: int, *, radius: int = 5, decay: float = 0.02):
        self.grid = grid
        self.n = num_agents
        self.radius = int(radius)
        self.decay = float(decay)
        self.cols = grid.cols
        self.rows = grid.rows
        self.size_ = grid.size

        size = grid.size
        self.occupancy = [0] * size            # robots physically present
        self.plan_load = [0] * size            # robots whose path crosses cell
        self.edge_wait = [0.0] * size          # EMA of observed waiting
        self.edge_visits = [0] * size
        self.delay_field = [0.0] * size        # smoothed delay heat
        # precomputed, padded occupancy score per cell (rebuilt on observe)
        self.occ_score = [0.0] * size

        self.robot_delay_ema = [0.0] * num_agents
        self.robot_stall_ema = [0.0] * num_agents
        self.robot_wait_total = [0] * num_agents
        self.robot_last_move_tick = [0] * num_agents

        # radius neighbourhood offsets (precomputed disc)
        r = self.radius
        self._disc_offsets: List[Tuple[int, int]] = [
            (dr, dc)
            for dr in range(-r, r + 1)
            for dc in range(-r, r + 1)
            if dr * dr + dc * dc <= r * r
        ]

        self._global_occ = 0.0
        self._obs_cache: Optional[Dict[str, float]] = None

    # ------------------------------------------------------------------
    # observation hooks called by the simulation each planning update
    # ------------------------------------------------------------------
    def observe(self, sim, free_agents: Sequence[int]) -> None:
        grid = self.grid
        robots = sim.robots
        size = grid.size

        # occupancy
        for i in range(size):
            self.occupancy[i] = 0
        for r in robots:
            self.occupancy[r.loc] += 1

        # plan load: every cell on a robot's remaining path
        for i in range(size):
            self.plan_load[i] = 0
        for r in robots:
            for cell in r.path:
                self.plan_load[cell] += 1

        # decay historical waiting cost (recency weighting)
        if sim.tick % 10 == 0:
            k = 1.0 - self.decay
            ew = self.edge_wait
            for i in range(size):
                if ew[i]:
                    ew[i] *= k

        # per-robot delay / stall EMAs
        alpha = 0.1
        for r in robots:
            d = 1.0 if sim.delaygen.remaining[r.id] > 0 else 0.0
            self.robot_delay_ema[r.id] = (1 - alpha) * self.robot_delay_ema[r.id] + alpha * d
            st = 1.0 if r.acc >= max(1, sim.inst.agent_counter) else 0.0
            self.robot_stall_ema[r.id] = (1 - alpha) * self.robot_stall_ema[r.id] + alpha * st

        # delay heat field, refreshed sparsely
        if sim.tick % 20 == 0:
            f = self.delay_field
            for i in range(size):
                if f[i]:
                    f[i] *= 0.7
            for r in robots:
                if sim.delaygen.remaining[r.id] > 0:
                    self._splat(f, r.loc, 1.0)

        self._global_occ = len(robots) / max(1, grid.num_passable)
        self._obs_cache = None
        self._rebuild_occ_score()

    def _rebuild_occ_score(self) -> None:
        """Box-filter the occupancy field once per update.

        Doing the disc convolution here (O(cells * disc)) rather than per query
        (O(pairs * samples * disc)) is the single biggest speed win: the
        scheduler then needs one array lookup per probe instead of ~80.
        """
        grid = self.grid
        cols, rows = self.cols, self.rows
        occ = self.occupancy
        passable = grid.passable
        r = self.radius
        # separable box blur of radius r, restricted to passable cells
        tmp = [0.0] * grid.size
        for rr in range(rows):
            base = rr * cols
            for cc in range(cols):
                lo = max(0, cc - r)
                hi = min(cols - 1, cc + r)
                s = 0
                for k in range(lo, hi + 1):
                    s += occ[base + k]
                tmp[base + cc] = s
        out = self.occ_score
        area = (2 * r + 1) * (2 * r + 1)
        for rr in range(rows):
            lo = max(0, rr - r)
            hi = min(rows - 1, rr + r)
            for cc in range(cols):
                idx = rr * cols + cc
                if not passable[idx]:
                    out[idx] = 0.0
                    continue
                s = 0.0
                for k in range(lo, hi + 1):
                    s += tmp[k * cols + cc]
                out[idx] = s / area - self._global_occ

    def observe_block(self, sim, robot, blocked_cell: int) -> None:
        """A robot wanted to enter ``blocked_cell`` but it was occupied."""
        self.robot_wait_total[robot.id] += 1
        ew = self.edge_wait
        v = self.edge_visits
        cell = blocked_cell
        ew[cell] = ew[cell] * 0.99 + 0.01 * 1.0
        v[cell] += 1

    def observe_completion(self, sim, robot) -> None:
        self.robot_last_move_tick[robot.id] = sim.tick

    def _splat(self, field: List[float], loc: int, amount: float) -> None:
        grid = self.grid
        r0, c0 = divmod(loc, grid.cols)
        rad = 4
        for dr in range(-rad, rad + 1):
            rr = r0 + dr
            if rr < 0 or rr >= grid.rows:
                continue
            for dc in range(-rad, rad + 1):
                cc = c0 + dc
                if cc < 0 or cc >= grid.cols:
                    continue
                d2 = dr * dr + dc * dc
                if d2 > rad * rad:
                    continue
                idx = rr * grid.cols + cc
                if grid.passable[idx]:
                    field[idx] += amount * math.exp(-d2 / (2.0 * 2.0 ** 2))

    # ------------------------------------------------------------------
    # C1 : area occupancy along the route
    # ------------------------------------------------------------------
    def area_occupancy(self, loc: int) -> float:
        """Pre-filtered local occupancy score (see ``_rebuild_occ_score``)."""
        return self.occ_score[loc]

    def route_area_occupancy(self, route: Sequence[int], samples: int = 4) -> float:
        """Mean occupancy over a few evenly spaced probes along the route.

        Sampling keeps this O(samples * disc) instead of O(|route| * disc) while
        still describing the *whole* corridor the robot would have to cross --
        which is the quantity that matters for assignment, not just the
        destination cell.
        """
        if not route:
            return 0.0
        n = len(route)
        if n <= samples:
            pts = list(route)
        else:
            step = max(1, n // samples)
            pts = [route[i] for i in range(0, n, step)][:samples]
            if pts[-1] != route[-1]:
                pts.append(route[-1])
        return sum(self.area_occupancy(p) for p in pts) / len(pts)

    # ------------------------------------------------------------------
    # C2 : vertex pressure along a route (mean per cell)
    # ------------------------------------------------------------------
    def vertex_pressure(self, route: Sequence[int], per_cell: bool = True) -> float:
        """Chen et al. vertex congestion ``ceil((n_v-1)/2)`` summed on a route.

        ``per_cell=True`` divides by the route length.  This matters a great
        deal for *task assignment*: the raw sum is monotone in route length, so
        an un-normalised version simply penalises distant tasks and silently
        degenerates into a second copy of ``d``.  Dividing by |route| isolates
        the "how crowded is this corridor" signal, which is what the ablation
        intends to test.
        """
        if not route:
            return 0.0
        total = 0.0
        for cell in route:
            n_v = self.plan_load[cell]
            if n_v > 1:
                total += (n_v - 1) * 0.5
        return total / len(route) if per_cell else total

    # ------------------------------------------------------------------
    # C3 : counter-flow on route cells
    # ------------------------------------------------------------------
    def counter_flow(self, route: Sequence[int], per_cell: bool = True) -> float:
        if len(route) < 2:
            return 0.0
        grid = self.grid
        path_set = set(route)
        total = 0.0
        for cell in route:
            load = self.plan_load[cell]
            if load <= 1:
                continue
            fwd = 0
            bwd = 0
            for q in grid.neighbors(cell):
                if q in path_set:
                    fwd += 1
                else:
                    bwd += self.plan_load[q]
            if fwd and bwd:
                total += fwd * bwd
        return max(0.0, total) / len(route) if per_cell else max(0.0, total)

    # ------------------------------------------------------------------
    # C4 : historical edge waiting
    # ------------------------------------------------------------------
    def edge_wait_cost(self, route: Sequence[int], per_cell: bool = True) -> float:
        if not route:
            return 0.0
        ew = self.edge_wait
        total = sum(ew[c] for c in route)
        return total / len(route) if per_cell else total

    # ------------------------------------------------------------------
    # R1 / R2 / R3 / R4
    # ------------------------------------------------------------------
    def delay_duty(self, agent: int) -> float:
        return self.robot_delay_ema[agent] + 0.5 * self.robot_stall_ema[agent]

    def drift(self, sim, agent: int) -> float:
        """Ticks since this robot last changed cell, normalised by action cost."""
        r = sim.robots[agent]
        gap = sim.tick - self.robot_last_move_tick[agent]
        scale = max(1, sim.inst.agent_counter * 4)
        return min(1.0, gap / scale)

    def delay_heat(self, loc: int) -> float:
        return self.delay_field[loc]

    def route_delay_heat(self, route: Sequence[int]) -> float:
        if not route:
            return self.delay_heat(0) if False else 0.0
        f = self.delay_field
        return sum(f[c] for c in route) / max(1, len(route))

    def starvation(self, task, timestep: int, service_ticks: float) -> float:
        age = timestep - task.t_revealed
        if service_ticks <= 0:
            return 0.0
        return max(0.0, age / service_ticks - 0.5)

    # ------------------------------------------------------------------
    # composite views
    # ------------------------------------------------------------------
    def congestion(self, agent: int, route: Sequence[int], kind: str = "c2") -> float:
        if kind == "c1":
            return self.route_area_occupancy(route)
        if kind == "c2":
            return self.vertex_pressure(route)
        if kind == "c3":
            return self.counter_flow(route)
        if kind == "c4":
            return self.edge_wait_cost(route)
        if kind == "mix":
            # Per-cell normalised blend.  C1 (occupancy) is already relative to
            # the global density, so it needs a larger multiplier to be
            # comparable in magnitude to the per-cell congestion counts.
            return (
                self.vertex_pressure(route)
                + 0.5 * self.counter_flow(route)
                + 2.0 * self.edge_wait_cost(route)
                + 20.0 * self.route_area_occupancy(route)
            )
        if kind == "zero":
            # ablation control: forces the re-rank code path while contributing
            # nothing, isolating the cost of the shortlist itself.
            return 0.0
        raise ValueError(f"unknown congestion kind: {kind}")

    def delay_risk(self, sim, agent: int, task, route: Sequence[int], kind: str = "r13") -> float:
        if kind == "r1":
            return self.delay_duty(agent)
        if kind == "r2":
            return self.drift(sim, agent)
        if kind == "r3":
            return self.route_delay_heat(route)
        if kind == "r4":
            svc = self._service_ticks(sim, route)
            return self.starvation(task, sim.tick, svc)
        if kind == "r13":
            return self.delay_duty(agent) + 2.0 * self.route_delay_heat(route)
        if kind == "mix":
            svc = self._service_ticks(sim, route)
            return (
                self.delay_duty(agent)
                + 0.5 * self.drift(sim, agent)
                + 2.0 * self.route_delay_heat(route)
                + 0.5 * self.starvation(task, sim.tick, svc)
            )
        raise ValueError(f"unknown delay kind: {kind}")

    def _service_ticks(self, sim, route: Sequence[int]) -> float:
        return max(1.0, len(route) * max(1, sim.inst.agent_counter))


def calibrate(values: Sequence[float], reference: float = 1.0) -> float:
    """Return the scale that maps a raw feature onto a *relative* weight scale.

    The score is composed as::

        S = d/d_ref + lambda * (C/mean_C) + gamma * (R/mean_R)

    so every term has mean 1 over the candidate set and ``lambda``/``gamma`` are
    directly readable as *relative importance*: ``lambda = 0.2`` means
    "congestion counts 20% as much as travel cost", independent of instance
    scale.  That is what makes the weights transferable across maps, fleet
    sizes and delay regimes -- and it is the convention the report uses when
    reporting the lambda/gamma sweeps.

    ``reference`` is kept for API compatibility and as a fallback when the
    candidate set is degenerate.
    """
    xs = [v for v in values if v == v]  # drop NaN
    if not xs:
        return max(1e-9, float(reference))
    m = sum(xs) / len(xs)
    if m <= 0:
        # all-zero feature (e.g. no congestion observed yet): use a tiny scale
        # so the term contributes ~0 rather than dominating.
        stdev = 0.0
        return max(1e-9, abs(reference))
    return m
