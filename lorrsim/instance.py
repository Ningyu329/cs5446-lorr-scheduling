"""Instance parsing for the LoRR 2026 Task Scheduling Track.

Implements the on-disk formats documented in the Start-Kit
``Input_Output_Format.md``:

  map file      ``type octile / height H / width W / map`` then H text rows
                with ``.`` free, ``@``/``T`` obstacle, ``E`` emitter, ``S`` service
  agent file    ``# comment`` optional, then N, then N linearised start cells
  task file     ``# comment`` optional, then M, then M rows of
                ``loc1,loc2,...`` (the ordered errand sequence of one task)
  instance JSON mapFile / agentFile / taskFile / teamSize / numTasksReveal /
                agentCounter / agentSize / delayConfig

Locations are linearised as ``row * width + col``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .grid import Grid

OBSTACLES = frozenset("@T")


@dataclass(frozen=True)
class DelayConfig:
    """Runtime delay generator configuration (mirrors Start-Kit ``DelayConfig``)."""

    seed: int = 0
    min_delay: int = 1
    max_delay: int = 4
    event_model: str = "bernoulli"  # bernoulli | poisson
    p_delay: float = 0.01
    duration_model: str = "uniform"  # uniform | gaussian
    gauss_mean_ratio: float = 0.5
    gauss_std_ratio: float = 0.22

    @classmethod
    def from_json(cls, obj: Optional[dict]) -> "DelayConfig":
        if not obj:
            return cls()
        return cls(
            seed=int(obj.get("seed", 0)),
            min_delay=int(obj.get("minDelay", 1)),
            max_delay=int(obj.get("maxDelay", 4)),
            event_model=str(obj.get("eventModel", "bernoulli")).lower(),
            p_delay=float(obj.get("pDelay", 0.01)),
            duration_model=str(obj.get("durationModel", "uniform")).lower(),
            gauss_mean_ratio=float(obj.get("gaussMeanRatio", 0.5)),
            gauss_std_ratio=float(obj.get("gaussStdRatio", 0.22)),
        )

    def to_json(self) -> dict:
        return {
            "seed": self.seed,
            "minDelay": self.min_delay,
            "maxDelay": self.max_delay,
            "eventModel": self.event_model,
            "pDelay": self.p_delay,
            "durationModel": self.duration_model,
            "gaussMeanRatio": self.gauss_mean_ratio,
            "gaussStdRatio": self.gauss_std_ratio,
        }

    def scaled(self, *, p_delay: Optional[float] = None, seed: Optional[int] = None) -> "DelayConfig":
        """Return a copy with delay intensity and/or seed overridden."""
        d = DelayConfig(**{**self.__dict__})
        if p_delay is not None:
            d = DelayConfig(**{**d.__dict__, "p_delay": float(p_delay)})
        if seed is not None:
            d = DelayConfig(**{**d.__dict__, "seed": int(seed)})
        return d


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return [ln.rstrip("\n").rstrip("\r") for ln in fh]


def _strip_comments(lines: Iterable[str]) -> List[str]:
    out = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        out.append(ln)
    return out


@dataclass
class Instance:
    """A fully-loaded LoRR instance."""

    name: str
    map_name: str
    grid: Grid
    agent_starts: List[int]
    tasks: List[Tuple[int, ...]]
    team_size: int
    num_tasks_reveal: float = 1.5
    agent_counter: int = 10
    agent_size: float = 1.0
    delay_config: DelayConfig = field(default_factory=DelayConfig)
    domain: str = ""
    source_path: str = ""

    # ---- derived -------------------------------------------------------
    @property
    def num_passable(self) -> int:
        return self.grid.num_passable

    @property
    def density(self) -> float:
        """Robots per passable cell -- the density axis of the study."""
        return self.team_size / max(1, self.num_passable)

    @property
    def num_errands(self) -> int:
        return sum(len(t) for t in self.tasks)

    def to_json(self) -> dict:
        return {
            "mapFile": f"maps/{self.map_name}.map",
            "teamSize": self.team_size,
            "numTasksReveal": self.num_tasks_reveal,
            "version": "2026 LoRR",
            "agentCounter": self.agent_counter,
            "agentSize": self.agent_size,
            "delayConfig": self.delay_config.to_json(),
        }

    def with_params(
        self,
        *,
        team_size: Optional[int] = None,
        delay_config: Optional[DelayConfig] = None,
        agent_counter: Optional[int] = None,
    ) -> "Instance":
        """Return a variant of this instance used for density/delay sweeps."""
        starts = list(self.agent_starts)
        ts = self.team_size if team_size is None else int(team_size)
        if ts > len(starts):
            raise ValueError(
                f"requested team_size={ts} exceeds available start cells ({len(starts)})"
            )
        return Instance(
            name=self.name,
            map_name=self.map_name,
            grid=self.grid,
            agent_starts=starts[:ts],
            tasks=list(self.tasks),
            team_size=ts,
            num_tasks_reveal=self.num_tasks_reveal,
            agent_counter=self.agent_counter if agent_counter is None else int(agent_counter),
            agent_size=self.agent_size,
            delay_config=self.delay_config if delay_config is None else delay_config,
            domain=self.domain,
            source_path=self.source_path,
        )


def load_map(path: str) -> Tuple[Grid, str]:
    lines = _read_lines(path)
    if not lines:
        raise ValueError(f"empty map file: {path}")
    height = width = None
    header_end = 0
    for i, ln in enumerate(lines):
        key = ln.strip().lower()
        if key.startswith("height"):
            height = int(ln.split()[1])
        elif key.startswith("width"):
            width = int(ln.split()[1])
        elif key == "map":
            header_end = i + 1
            break
    if height is None or width is None:
        raise ValueError(f"map file missing height/width: {path}")
    rows = lines[header_end : header_end + height]
    if len(rows) != height:
        raise ValueError(f"map file declares height={height} but has {len(rows)} rows: {path}")
    passable = [False] * (height * width)
    for r, row in enumerate(rows):
        if len(row) < width:
            row = row.ljust(width, "@")
        base = r * width
        for c in range(width):
            passable[base + c] = row[c] not in OBSTACLES
    map_name = os.path.splitext(os.path.basename(path))[0]
    return Grid(rows=height, cols=width, passable=passable), map_name


def load_agents(path: str) -> List[int]:
    lines = _strip_comments(_read_lines(path))
    if not lines:
        raise ValueError(f"empty agent file: {path}")
    n = int(lines[0].split()[0])
    starts: List[int] = []
    for ln in lines[1:]:
        for tok in ln.replace(",", " ").split():
            starts.append(int(tok))
            if len(starts) >= n:
                break
        if len(starts) >= n:
            break
    if len(starts) < n:
        raise ValueError(f"agent file claims {n} agents but only {len(starts)} parsed: {path}")
    return starts[:n]


def load_tasks(path: str) -> List[Tuple[int, ...]]:
    lines = _strip_comments(_read_lines(path))
    if not lines:
        raise ValueError(f"empty task file: {path}")
    m = int(lines[0].split()[0])
    tasks: List[Tuple[int, ...]] = []
    for ln in lines[1:]:
        locs = tuple(int(tok) for tok in ln.replace(",", " ").split() if tok.strip())
        if not locs:
            continue
        tasks.append(locs)
        if len(tasks) >= m:
            break
    if len(tasks) < m:
        raise ValueError(f"task file claims {m} tasks but only {len(tasks)} parsed: {path}")
    return tasks[:m]


def load_instance(path: str, *, name: Optional[str] = None) -> Instance:
    """Load an instance JSON, resolving map/agent/task paths relative to it."""
    base = os.path.dirname(os.path.abspath(path))
    with open(path, "r", encoding="utf-8") as fh:
        spec = json.load(fh)

    grid, map_name = load_map(os.path.join(base, spec["mapFile"]))
    starts = load_agents(os.path.join(base, spec["agentFile"]))
    tasks = load_tasks(os.path.join(base, spec["taskFile"]))
    team_size = int(spec.get("teamSize", len(starts)))

    st = os.path.abspath(path)
    # instance JSONs live in <domain>/<name>.json -> domain is the parent dir
    parent = os.path.basename(base)
    domain = parent if parent.endswith(".domain") else parent

    return Instance(
        name=name or os.path.splitext(os.path.basename(path))[0],
        map_name=map_name,
        grid=grid,
        agent_starts=starts,
        tasks=tasks,
        team_size=team_size,
        num_tasks_reveal=float(spec.get("numTasksReveal", 1.5)),
        agent_counter=int(spec.get("agentCounter", 10)),
        agent_size=float(spec.get("agentSize", 1.0)),
        delay_config=DelayConfig.from_json(spec.get("delayConfig")),
        domain=domain or "unknown",
        source_path=st,
    )


def discover_instances(root: str) -> List[str]:
    """Find every instance JSON under ``root`` (any ``.json`` with mapFile)."""
    found: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in sorted(filenames):
            if not fn.endswith(".json"):
                continue
            p = os.path.join(dirpath, fn)
            try:
                with open(p, "r", encoding="utf-8") as fh:
                    head = fh.read(4096)
                if '"mapFile"' in head and '"agentFile"' in head:
                    found.append(p)
            except OSError:
                continue
    return sorted(found)


def instance_table(paths: Sequence[str]) -> List[Dict[str, object]]:
    """Summary rows (no grid retained) for reporting and config building."""
    rows = []
    for p in paths:
        inst = load_instance(p)
        rows.append(
            {
                "name": inst.name,
                "domain": inst.domain,
                "team_size": inst.team_size,
                "map": inst.map_name,
                "rows": inst.grid.rows,
                "cols": inst.grid.cols,
                "passable": inst.num_passable,
                "density": round(inst.density, 4),
                "num_tasks": len(inst.tasks),
                "num_errands": inst.num_errands,
                "mean_errands": round(inst.num_errands / max(1, len(inst.tasks)), 2),
                "agent_counter": inst.agent_counter,
                "delay_p": inst.delay_config.p_delay,
                "delay_event": inst.delay_config.event_model,
            }
        )
    return rows
