"""Static grid geometry and BFS distance tables.

The official Start-Kit uses a heuristic table (``DefaultPlanner::get_h``) for
task-assignment distances and 4-connected movement in the planner.  We mirror
that: the *task-assignment* distance ``d`` is a 4-connected BFS distance
counted in **actions**, which the schedulers then convert to ticks via
``agentCounter`` so that all score terms share one unit.
"""

from __future__ import annotations

import array
from collections import OrderedDict, deque
from typing import List, Optional, Sequence, Tuple

NEIGHBORS4: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))


class Grid:
    """4-connected grid with a bounded, LRU-cached BFS distance table.

    The cache is deliberately capacity-limited.  An unbounded per-cell cache
    costs ``O(cells)`` Python ints per entry -- on a 70,000-cell warehouse map
    with 2,500 robots that is multiple gigabytes, which is enough to OOM the
    process.  Task-assignment distances only re-use a small working set (the
    free robots' current cells plus errand cells), so an LRU of a few hundred
    entries captures nearly all the benefit at a fixed memory cost.
    """

    __slots__ = ("rows", "cols", "passable", "_num_passable", "_cache", "_max_cache", "max_cache_entries")

    def __init__(self, rows: int, cols: int, passable: Sequence[bool], max_cache_entries: int = 192):
        self.rows = int(rows)
        self.cols = int(cols)
        self.passable = list(passable)
        self._num_passable = sum(1 for p in self.passable if p)
        self._cache = OrderedDict()  # start cell -> array('i') distance map
        self._max_cache = {}
        self.max_cache_entries = int(max_cache_entries)

    # ---- basic ---------------------------------------------------------
    @property
    def size(self) -> int:
        return self.rows * self.cols

    @property
    def num_passable(self) -> int:
        return self._num_passable

    def is_passable(self, loc: int) -> bool:
        return 0 <= loc < len(self.passable) and self.passable[loc]

    def rc(self, loc: int) -> Tuple[int, int]:
        return divmod(loc, self.cols)

    def loc(self, r: int, c: int) -> int:
        return r * self.cols + c

    def neighbors(self, loc: int) -> List[int]:
        r, c = divmod(loc, self.cols)
        out: List[int] = []
        p = self.passable
        cols = self.cols
        if r > 0:
            q = loc - cols
            if p[q]:
                out.append(q)
        if r + 1 < self.rows:
            q = loc + cols
            if p[q]:
                out.append(q)
        if c > 0:
            q = loc - 1
            if p[q]:
                out.append(q)
        if c + 1 < self.cols:
            q = loc + 1
            if p[q]:
                out.append(q)
        return out

    # ---- distances -----------------------------------------------------
    def bfs(self, start: int) -> "array.array":
        """Single-source BFS over passable cells; unreachable = -1.

        Returns a compact ``array('i')`` (4 bytes/cell vs ~36 for a Python int
        list).  On the 70,000-cell warehouse map this is the difference between
        70 MB and 645 MB of cache.  Returned arrays are shared -- treat as
        read-only.
        """
        cached = self._cache.get(start)
        if cached is not None:
            self._cache.move_to_end(start)
            return cached
        dist = array.array("i", [-1]) * self.size
        if self.is_passable(start):
            dist[start] = 0
            dq = deque([start])
            cols = self.cols
            p = self.passable
            rows = self.rows
            while dq:
                cur = dq.popleft()
                nd = dist[cur] + 1
                r, c = divmod(cur, cols)
                if r > 0:
                    q = cur - cols
                    if p[q] and dist[q] < 0:
                        dist[q] = nd
                        dq.append(q)
                if r + 1 < rows:
                    q = cur + cols
                    if p[q] and dist[q] < 0:
                        dist[q] = nd
                        dq.append(q)
                if c > 0:
                    q = cur - 1
                    if p[q] and dist[q] < 0:
                        dist[q] = nd
                        dq.append(q)
                if c + 1 < cols:
                    q = cur + 1
                    if p[q] and dist[q] < 0:
                        dist[q] = nd
                        dq.append(q)
        self._cache[start] = dist
        if len(self._cache) > self.max_cache_entries:
            self._cache.popitem(last=False)
        return dist

    def dist(self, a: int, b: int) -> int:
        """4-connected grid distance; ``-1`` when unreachable."""
        if a == b:
            return 0
        return self.bfs(a)[b]

    def dist_or_inf(self, a: int, b: int, default: int = 10 ** 6) -> int:
        d = self.dist(a, b)
        return default if d < 0 else d

    def max_dist(self, start: int) -> int:
        """Eccentricity of ``start`` -- useful for normalising scores."""
        if start in self._max_cache:
            return self._max_cache[start]
        d = self.bfs(start)
        m = max((x for x in d if x >= 0), default=0)
        self._max_cache[start] = m
        return m

    def clear_cache(self) -> None:
        self._cache.clear()
        self._max_cache.clear()

    # ---- convenience for reporting -------------------------------------
    def render(self, marks: Optional[dict] = None, max_rows: int = 60) -> str:
        """ASCII render (optionally marking cells) for docs/debug output."""
        marks = marks or {}
        out = []
        for r in range(min(self.rows, max_rows)):
            row = []
            for c in range(self.cols):
                loc = r * self.cols + c
                if loc in marks:
                    row.append(str(marks[loc])[:1])
                else:
                    row.append("." if self.passable[loc] else "@")
            out.append("".join(row))
        if self.rows > max_rows:
            out.append(f"... ({self.rows - max_rows} more rows)")
        return "\n".join(out)
