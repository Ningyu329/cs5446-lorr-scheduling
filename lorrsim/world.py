"""Tick-based lifelong multi-robot simulation.

This is a purpose-built but behaviourally faithful stand-in for the Start-Kit
``CompetitionSystem`` / ``Simulator`` loop, scoped to what a *task-scheduling*
study needs:

* two-rate control: slow *planning updates* (assignment) and fast *execution
  ticks* (motion, delay, contention),
* multi-tick actions via ``agentCounter``,
* per-tick stochastic delays (``DelayGenerator``),
* cell-occupancy contention with persistent robot priorities, which produces
  the corridor queueing that makes congestion-aware assignment meaningful,
* task lifecycle exactly as in ``TaskManager`` (reveal -> open -> closed).

All schedulers share this loop verbatim; only the assignment policy changes.
"""

from __future__ import annotations

import random
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Tuple

from .delay import DelayGenerator
from .features import FeatureBank
from .instance import Instance
from .tasks import TaskManager


@dataclass
class Robot:
    id: int
    loc: int
    task_id: int = -1
    path: Deque[int] = field(default_factory=lambda: deque())
    acc: int = 0
    priority: int = 0
    waiting_ticks: int = 0
    moves: int = 0
    delay_ticks: int = 0
    tasks_done: int = 0
    last_move_tick: int = 0
    last_progress_tick: int = 0


@dataclass
class SimResult:
    instance: str
    scheduler: str
    seed: int
    team_size: int
    density: float
    delay_p: float
    delay_event: str
    agent_counter: int
    ticks: int
    num_task_finished: int
    throughput: float
    mean_wait: float
    p95_wait: float
    max_wait: float
    load_cv: float
    load_gini: float
    total_delay_ticks: int
    mean_moves_per_robot: float
    sched_calls: int
    sched_ms_mean: float
    sched_ms_p95: float
    sched_ms_max: float
    sched_budget_hit: int
    assign_attempts: int
    extra: Dict[str, float] = field(default_factory=dict)

    def as_row(self) -> Dict[str, object]:
        d = {
            "instance": self.instance,
            "scheduler": self.scheduler,
            "seed": self.seed,
            "team_size": self.team_size,
            "density": round(self.density, 4),
            "delay_p": self.delay_p,
            "delay_event": self.delay_event,
            "agent_counter": self.agent_counter,
            "ticks": self.ticks,
            "num_task_finished": self.num_task_finished,
            "throughput": round(self.throughput, 6),
            "mean_wait": round(self.mean_wait, 3),
            "p95_wait": round(self.p95_wait, 3),
            "max_wait": round(self.max_wait, 3),
            "load_cv": round(self.load_cv, 4),
            "load_gini": round(self.load_gini, 4),
            "total_delay_ticks": self.total_delay_ticks,
            "mean_moves_per_robot": round(self.mean_moves_per_robot, 3),
            "sched_calls": self.sched_calls,
            "sched_ms_mean": round(self.sched_ms_mean, 4),
            "sched_ms_p95": round(self.sched_ms_p95, 4),
            "sched_ms_max": round(self.sched_ms_max, 4),
            "sched_budget_hit": self.sched_budget_hit,
            "assign_attempts": self.assign_attempts,
        }
        for k, v in self.extra.items():
            d[k] = round(v, 6) if isinstance(v, float) else v
        return d


def _gini(values: Sequence[int]) -> float:
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0:
        return 0.0
    s = sum(xs)
    if s <= 0:
        return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2.0 * cum) / (n * s) - (n + 1.0) / n


def _p95(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(0.95 * (len(s) - 1))))
    return float(s[idx])


class Simulation:
    """Runs one (instance, scheduler, seed) episode."""

    def __init__(
        self,
        instance: Instance,
        scheduler,
        *,
        seed: int = 0,
        horizon: int = 1200,
        plan_comm: int = 20,
        sched_budget_ms: float = 50.0,
        collect_trace: bool = False,
    ):
        self.inst = instance
        self.grid = instance.grid
        self.scheduler = scheduler
        self.seed = int(seed)
        self.horizon = int(horizon)
        self.plan_comm = int(plan_comm)
        self.sched_budget_ms = float(sched_budget_ms)
        self.collect_trace = bool(collect_trace)

        # One seeded RNG drives *everything* stochastic in the episode: robot
        # relocation, tie-breaking order and the delay generator.  Without
        # forwarding the run seed into the delay config, every seed would draw
        # an identical delay stream and the seeds would be indistinguishable.
        rng = random.Random(self.seed ^ 0x5EED)
        inst_dc = instance.delay_config
        scaled_dc = inst_dc.scaled(seed=(inst_dc.seed * 1000003 + self.seed) & 0x7FFFFFFF)
        n = instance.team_size
        start_cells = instance.agent_starts[:n]
        # robots must start on distinct passable cells
        used = set()
        self.robots: List[Robot] = []
        for i, loc in enumerate(start_cells):
            if loc in used or not self.grid.is_passable(loc):
                # relocate deterministically to first free passable cell
                loc = self._first_free(used)
            used.add(loc)
            self.robots.append(
                Robot(id=i, loc=loc, priority=rng.randrange(1 << 30))
            )
        self.occupant: List[int] = [-1] * self.grid.size
        for r in self.robots:
            self.occupant[r.loc] = r.id

        self.episode_instance = instance.with_params(delay_config=scaled_dc)
        self.taskmgr = TaskManager(
            instance.tasks, n, num_tasks_reveal=instance.num_tasks_reveal
        )
        self.delaygen = DelayGenerator(scaled_dc, n)
        self.features = FeatureBank(self.grid, n)
        self.tick = 0
        self._sched_ms: List[float] = []
        self._budget_hits = 0
        self._assign_attempts = 0
        self.trace: List[Tuple[int, int]] = []  # (tick, cumulative finished)

    # ---- setup helpers --------------------------------------------------
    def _first_free(self, used: set) -> int:
        for loc in range(self.grid.size):
            if self.grid.passable[loc] and loc not in used:
                return loc
        raise RuntimeError("no free passable cell available")

    # ---- path handling --------------------------------------------------
    def _goal_of(self, robot: Robot) -> int:
        """Current errand cell the robot must reach, or -1."""
        if robot.task_id < 0:
            return -1
        t = self.taskmgr.pool.get(robot.task_id)
        if t is None or t.is_finished:
            return -1
        return t.next_loc

    def _retarget(self, robot: Robot) -> None:
        """Record the intended route (used for congestion features only).

        Execution itself follows the distance map greedily each tick, which is
        what keeps the loop robust under contention.
        """
        robot.path.clear()
        robot.acc = 0
        goal = self._goal_of(robot)
        if goal < 0 or goal == robot.loc:
            return
        dmap = self.grid.bfs(goal)
        cur = robot.loc
        guard = 0
        while cur != goal and guard < self.grid.size * 4:
            guard += 1
            nd = dmap[cur]
            if nd <= 0:
                break
            nxt = -1
            for q in self.grid.neighbors(cur):
                if dmap[q] == nd - 1:
                    nxt = q
                    break
            if nxt < 0:
                break
            robot.path.append(nxt)
            cur = nxt

    # ---- main loop ------------------------------------------------------
    def run(self) -> SimResult:
        inst = self.inst
        taskmgr = self.taskmgr
        taskmgr.initialize(self.tick)
        self.scheduler.reset(self)

        # initial assignment
        self._planning_update()

        ac = max(1, inst.agent_counter)
        while self.tick < self.horizon:
            self.tick += 1
            self._execution_tick(ac)

            if self.tick % self.plan_comm == 0:
                self._planning_update()

            if self.collect_trace and self.tick % 25 == 0:
                self.trace.append((self.tick, taskmgr.finished_count))

            if not taskmgr.pool and taskmgr._next_stream_idx >= len(taskmgr._stream):
                break

        self.delaygen.flush(self.tick)
        return self._summarise()

    def _planning_update(self) -> None:
        taskmgr = self.taskmgr
        taskmgr.top_up(self.tick)

        # free robots: no task, or their open task just closed
        free = []
        for r in self.robots:
            if r.task_id < 0:
                free.append(r.id)
                continue
            t = taskmgr.pool.get(r.task_id)
            if t is None:
                r.task_id = -1
                r.path.clear()
                free.append(r.id)
            elif t.agent_assigned != r.id:
                t.agent_assigned = r.id

        # refresh observed/derived state used by congestion & risk features
        self.features.observe(self, free)

        if not free:
            self._refresh_paths()
            return

        schedule = list(self._current_schedule())
        n = len(self.robots)
        if len(schedule) < n:
            schedule.extend([-1] * (n - len(schedule)))

        t0 = time.perf_counter()
        proposed = self.scheduler.plan(self, free, schedule, self.sched_budget_ms)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self._sched_ms.append(dt_ms)
        if dt_ms > self.sched_budget_ms:
            self._budget_hits += 1
        self._assign_attempts += len(free)

        if proposed is None:
            self._refresh_paths()
            return

        self._apply_schedule(proposed)
        self._refresh_paths()

    def _current_schedule(self) -> List[int]:
        return [r.task_id for r in self.robots]

    def _apply_schedule(self, proposed: Sequence[int]) -> None:
        """Validate and commit a proposed schedule (official semantics)."""
        taskmgr = self.taskmgr
        n = len(self.robots)
        sched = list(proposed[:n])
        if len(sched) < n:
            sched.extend([-1] * (n - len(sched)))
        for i, tid in enumerate(sched):
            if tid is None:
                sched[i] = -1
                continue
            tid = int(tid)
            sched[i] = tid
            if tid < 0:
                continue
            if not taskmgr.is_assignable(tid, i, sched):
                sched[i] = -1

        for i in range(n):
            r = self.robots[i]
            tid = sched[i]
            if tid >= 0 and tid != r.task_id:
                if r.task_id >= 0:
                    taskmgr.unbind(r.task_id)
                r.task_id = tid
                taskmgr.bind(tid, i)
                r.path.clear()
                r.acc = 0
            elif tid < 0 and r.task_id >= 0:
                # official rule: never drop an already-opened task
                t = taskmgr.pool.get(r.task_id)
                if t is None:
                    r.task_id = -1
                elif not t.is_opened:
                    taskmgr.unbind(r.task_id)
                    r.task_id = -1
                    r.path.clear()

    def _refresh_paths(self, robot: Optional[Robot] = None) -> None:
        """Ensure each robot has a plan to its current errand."""
        robots = [robot] if robot is not None else self.robots
        for r in robots:
            if r.task_id < 0:
                r.path.clear()
                r.acc = 0
                continue
            t = self.taskmgr.pool.get(r.task_id)
            if t is None:
                r.task_id = -1
                r.path.clear()
                continue
            if r.loc == t.next_loc:
                self._arrive(r, t.next_loc)
                continue
            if not r.path or r.path[-1] != t.next_loc:
                self._retarget(r)
            elif r.path and r.path[0] == r.loc:
                r.path.popleft()

    def _arrive(self, r: Robot, loc: int) -> None:
        if r.task_id < 0:
            return
        done = self.taskmgr.advance_task(r.task_id, loc, r.id, self.tick)
        if done is not None:
            r.tasks_done += 1
            r.task_id = -1
            r.path.clear()
            self.features.observe_completion(self, r)
        else:
            self._retarget(r)

    def _execution_tick(self, agent_counter: int) -> None:
        """One execution tick with PIBT-style recursive priority inheritance.

        Movement is resolved in a rotating fair order.  When robot ``r`` wants
        the cell currently held by ``j``, we first try to move ``j`` (and, if
        needed, whoever blocks ``j``) recursively, depth-limited.  This is the
        *priority inheritance with backtracking* idea used by the official
        default planner, and it is what lets convoys flow instead of
        deadlocking: a follower moves in the same tick as its leader.

        A robot that still cannot move simply yields -- which is exactly how
        the official overlap model converts an unsafe action into a wait, so
        corridor queueing (the phenomenon the study is about) still emerges.
        """
        grid = self.grid
        occupant = self.occupant
        ac = agent_counter

        movers = [r for r in self.robots if r.path and r.loc != r.path[0]]
        if not movers:
            return

        available = [False] * len(self.robots)
        for r in movers:
            available[r.id] = True
        delayed = self.delaygen.tick(available, self.tick)

        n = len(movers)
        start = self.tick % n
        ordered = movers[start:] + movers[:start]

        self._acted = [False] * len(self.robots)
        self._visited = [False] * len(self.robots)
        self._ac = ac
        self._delayed = delayed

        for r in ordered:
            if self._acted[r.id]:
                continue
            self._try_move(r, depth=0)

        # Stuck detection: a robot that has not changed cell for a long time is
        # in a local deadlock (typically a cycle of robots each wanting the
        # next robot's cell).  Give it a lateral escape -- a deliberate
        # substitute for the planner's job of avoiding such cycles, keeping the
        # measurement of *scheduling* effects uncontaminated by lock-ups.
        self._resolve_stuck(max(6, 3 * ac))

    def _resolve_stuck(self, stuck_window: int) -> None:
        grid = self.grid
        occupant = self.occupant
        for r in self.robots:
            if not r.path or r.loc == r.path[0]:
                continue
            if self.tick - r.last_progress_tick < stuck_window:
                continue
            # try a lateral move to any free neighbouring cell that does not
            # increase our distance to the goal
            goal = r.path[-1]
            dmap = grid.bfs(goal)
            cur_d = dmap[r.loc]
            options = []
            for q in grid.neighbors(r.loc):
                if occupant[q] != -1:
                    continue
                dq = dmap[q]
                if dq >= 0 and (cur_d < 0 or dq <= cur_d + 1):
                    options.append((dq, q))
            if not options:
                # as a last resort allow any free neighbour
                options = [(0, q) for q in grid.neighbors(r.loc) if occupant[q] == -1]
            if not options:
                continue
            options.sort()
            _dq, q = options[0]
            occupant[r.loc] = -1
            occupant[q] = r.id
            r.loc = q
            r.moves += 1
            r.acc = 0
            r.last_progress_tick = self.tick
            r.last_move_tick = self.tick
            r.waiting_ticks += 1
            self._refresh_paths(r)

    # -- recursive movement ------------------------------------------------
    def _try_move(self, r: Robot, depth: int) -> bool:
        """Attempt to move ``r`` one cell, recursively unblocking first."""
        if self._acted[r.id]:
            return False
        if depth > 24:
            return False
        if self._delayed[r.id]:
            r.delay_ticks += 1
            self._acted[r.id] = True
            return False
        if not r.path:
            self._acted[r.id] = True
            return False

        r.acc += 1
        if r.acc < self._ac:
            return False  # not ready yet; may still advance next tick

        nxt = r.path[0]
        holder = self.occupant[nxt]
        if holder != -1 and holder != r.id:
            # priority inheritance: try to clear the way first
            if self._visited[r.id]:
                r.waiting_ticks += 1
                self.features.observe_block(self, r, nxt)
                self._acted[r.id] = True
                return False
            self._visited[r.id] = True
            mover_ahead = self.robots[holder]
            moved_ahead = False
            if not self._acted[mover_ahead.id]:
                moved_ahead = self._try_move(mover_ahead, depth + 1)
            self._visited[r.id] = False
            if not moved_ahead and self.occupant[nxt] != -1:
                r.waiting_ticks += 1
                self.features.observe_block(self, r, nxt)
                self._acted[r.id] = True
                return False

        if self.occupant[nxt] != -1:
            r.waiting_ticks += 1
            self._acted[r.id] = True
            return False

        self._move(r, nxt)
        self._acted[r.id] = True
        return True

    def _move(self, r: Robot, nxt: int) -> None:
        self.occupant[r.loc] = -1
        self.occupant[nxt] = r.id
        r.loc = nxt
        r.path.popleft()
        r.acc = 0
        r.moves += 1
        r.last_move_tick = self.tick
        r.last_progress_tick = self.tick
        t = self.taskmgr.pool.get(r.task_id) if r.task_id >= 0 else None
        if t is not None and t.next_loc == r.loc:
            self._arrive(r, r.loc)

    def _break_deadlock(self, blocked: List[Robot]) -> None:
        """Escape a fully jammed tick.

        Two mechanisms, both consistent with the official model's "unsafe
        action is overridden to a wait", but applied globally so the system
        still makes progress:

        1. step-aside: if a blocked robot has a free neighbouring cell that is
           not its goal, it shifts there and immediately re-plans.
        2. priority ageing: shuffle the acting order so a different set of
           robots gets first claim next tick.
        """
        grid = self.grid
        occupant = self.occupant
        for r in blocked:
            if not r.path:
                continue
            goal_cell = r.path[-1]
            for q in grid.neighbors(r.loc):
                if q == goal_cell or occupant[q] != -1:
                    continue
                self.occupant[r.loc] = -1
                self.occupant[q] = r.id
                r.loc = q
                r.path.clear()          # force re-plan from the new cell
                r.acc = max(1, self.inst.agent_counter) - 1
                r.waiting_ticks += 1
                self._refresh_paths(r)
                return
        # last resort: rotate priorities so the jam resolves within a few ticks
        for r in self.robots:
            r.priority = (r.priority * 1103515245 + 12345) & 0x7FFFFFFF

    # ---- summary --------------------------------------------------------
    def _summarise(self) -> SimResult:
        inst = self.inst
        taskmgr = self.taskmgr
        done_counts = [r.tasks_done for r in self.robots]
        ticks = max(1, self.tick)
        finished = taskmgr.finished_count
        mean_moves = sum(r.moves for r in self.robots) / max(1, len(self.robots))

        # per-task waiting time = completion tick - reveal tick
        waits = [c - rev for (c, _tid, _a, rev) in taskmgr.episode_finished]
        mean_wait = sum(waits) / len(waits) if waits else 0.0
        mc = sum(done_counts) / max(1, len(done_counts))
        var = sum((x - mc) ** 2 for x in done_counts) / max(1, len(done_counts))
        cv = (var ** 0.5) / mc if mc > 0 else 0.0

        sched = sorted(self._sched_ms)

        return SimResult(
            instance=inst.name,
            scheduler=getattr(self.scheduler, "name", type(self.scheduler).__name__),
            seed=self.seed,
            team_size=inst.team_size,
            density=inst.density,
            delay_p=inst.delay_config.p_delay,
            delay_event=inst.delay_config.event_model,
            agent_counter=inst.agent_counter,
            ticks=self.tick,
            num_task_finished=finished,
            throughput=finished / ticks,
            mean_wait=mean_wait,
            p95_wait=_p95(waits),
            max_wait=max(waits) if waits else 0.0,
            load_cv=cv,
            load_gini=_gini(done_counts),
            total_delay_ticks=self.delaygen.total_delayed_ticks(),
            mean_moves_per_robot=mean_moves,
            sched_calls=len(self._sched_ms),
            sched_ms_mean=(sum(sched) / len(sched)) if sched else 0.0,
            sched_ms_p95=_p95(sched),
            sched_ms_max=max(sched) if sched else 0.0,
            sched_budget_hit=self._budget_hits,
            assign_attempts=self._assign_attempts,
            extra={
                "wait_ticks_total": float(taskmgr.total_wait_ticks),
                "tasks_revealed": float(taskmgr._next_stream_idx),
            },
        )

    def _task_waits(self, taskmgr) -> List[int]:
        """Per-task (complete - reveal) using the reveal table."""
        return [c - rev for (c, _tid, _a, rev) in taskmgr.episode_finished]
