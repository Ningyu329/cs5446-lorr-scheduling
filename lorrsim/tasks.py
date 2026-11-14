"""Task pool and task manager, mirroring Start-Kit ``Tasks.h`` / ``TaskManager``.

Key semantics preserved from the official implementation:

* A ``task`` is an *ordered* sequence of errands (locations).
* A task becomes **open** once the assigned robot completes errand 0
  (``idx_next_loc >= 1``).  From then on it may not be re-assigned.
* A task is **closed** when all errands are completed; it leaves the pool.
* The pool always keeps ``numTasksReveal * teamSize`` revealed tasks
  available: every completion reveals one more task from the stream.
* Each robot holds at most one open task.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple


@dataclass
class Task:
    task_id: int
    locations: Tuple[int, ...]
    t_revealed: int = 0
    t_completed: int = -1
    agent_assigned: int = -1
    idx_next_loc: int = 0

    @property
    def next_loc(self) -> int:
        return self.locations[self.idx_next_loc]

    @property
    def is_finished(self) -> bool:
        return self.idx_next_loc >= len(self.locations)

    @property
    def is_opened(self) -> bool:
        """Opened == first errand completed == no longer re-assignable."""
        return self.idx_next_loc >= 1

    @property
    def num_errands(self) -> int:
        return len(self.locations)


class TaskManager:
    """Owns the revealed task stream and the live task pool."""

    def __init__(
        self,
        all_tasks: Sequence[Tuple[int, ...]],
        team_size: int,
        num_tasks_reveal: float = 1.5,
    ):
        self._stream: List[Tuple[int, ...]] = list(all_tasks)
        self._next_stream_idx = 0
        self.pool: Dict[int, Task] = {}
        self.team_size = int(team_size)
        self.num_tasks_reveal = float(num_tasks_reveal)
        self.target_pool_size = max(1, int(round(self.num_tasks_reveal * self.team_size)))
        self._next_task_id = 0
        self.new_tasks: List[int] = []
        self.finished_count = 0
        self.episode_finished: List[Tuple[int, int, int, int]] = []  # (tick, task_id, agent, reveal)
        self.robot_tasks_done: List[int] = [0] * self.team_size
        self.total_wait_ticks = 0

    # ---- reveal --------------------------------------------------------
    def _reveal_one(self) -> Optional[Task]:
        if self._next_stream_idx >= len(self._stream):
            return None
        locs = self._stream[self._next_stream_idx]
        self._next_stream_idx += 1
        t = Task(task_id=self._next_task_id, locations=tuple(locs))
        self._next_task_id += 1
        self.pool[t.task_id] = t
        return t

    def initialize(self, timestep: int = 0) -> None:
        """Fill the pool to ``numTasksReveal * teamSize`` and mark all new."""
        self.new_tasks = []
        while len(self.pool) < self.target_pool_size:
            t = self._reveal_one()
            if t is None:
                break
            t.t_revealed = timestep
            self.new_tasks.append(t.task_id)

    def top_up(self, timestep: int) -> None:
        """Keep ``numTasksReveal * teamSize`` tasks revealed, as the controller does."""
        self.new_tasks = []
        while len(self.pool) < self.target_pool_size:
            t = self._reveal_one()
            if t is None:
                break
            t.t_revealed = timestep
            self.new_tasks.append(t.task_id)

    # ---- assignment validity ------------------------------------------
    def is_assignable(self, task_id: int, agent: int, schedule: Sequence[int]) -> bool:
        """Mirror the official validity rules for a single (task, agent) pair."""
        t = self.pool.get(task_id)
        if t is None or t.is_finished:
            return False
        if t.is_opened:  # already started by someone -> frozen
            return t.agent_assigned == agent
        if t.agent_assigned != -1 and t.agent_assigned != agent:
            return False
        # cannot hand the same task to two robots in one schedule
        for other, tid in enumerate(schedule):
            if other != agent and tid == task_id:
                return False
        return True

    def bind(self, task_id: int, agent: int) -> None:
        t = self.pool.get(task_id)
        if t is None:
            return
        t.agent_assigned = agent

    def unbind(self, task_id: int) -> None:
        t = self.pool.get(task_id)
        if t is not None and not t.is_opened:
            t.agent_assigned = -1

    # ---- progress ------------------------------------------------------
    def advance_task(self, task_id: int, location: int, agent: int, timestep: int) -> Optional[int]:
        """Register arrival of ``agent`` at ``location`` for ``task_id``.

        Returns ``task_id`` when that arrival completed the task, else ``None``.
        """
        t = self.pool.get(task_id)
        if t is None or t.is_finished:
            return None
        if t.locations[t.idx_next_loc] != location:
            return None
        t.idx_next_loc += 1
        if t.is_finished:
            t.t_completed = timestep
            self.finished_count += 1
            self.robot_tasks_done[agent] += 1
            self.episode_finished.append((timestep, task_id, agent, t.t_revealed))
            self.total_wait_ticks += timestep - t.t_revealed
            del self.pool[task_id]
            return task_id
        return None

    # ---- helpers -------------------------------------------------------
    def free_tasks(self) -> List[int]:
        """Tasks that can still be handed to an unassigned robot."""
        return [tid for tid, t in self.pool.items() if not t.is_opened and t.agent_assigned == -1]

    def open_task_count(self) -> int:
        return sum(1 for t in self.pool.values() if t.is_opened)

    def backlog_wait(self, timestep: int) -> float:
        """Mean age of unassigned tasks -- feeds adaptive lambda."""
        un = [timestep - t.t_revealed for t in self.pool.values() if t.agent_assigned == -1]
        if not un:
            return 0.0
        return sum(un) / len(un)

    def mean_wait(self) -> float:
        if self.finished_count == 0:
            return 0.0
        return self.total_wait_ticks / self.finished_count
