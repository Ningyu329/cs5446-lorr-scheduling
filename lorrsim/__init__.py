"""lorrsim -- a faithful, self-contained Python re-implementation of the
League of Robot Runners (LoRR) 2026 *Task Scheduling Track* evaluation loop.

The official Start-Kit is C++ (CMake + Boost + pybind11).  This package
reproduces the parts of that loop that matter for a *task-scheduling* study:

    parsing        - map / agents / tasks / instance-JSON formats
    grid           - 4-connected grid with BFS distance tables
    delay          - bernoulli/poisson event model + uniform/gaussian duration
    simulation     - tick-based lifelong loop, counters, wait-on-delay
    planner        - a fixed, shared, distance-guided reference planner
    schedulers     - S0 / S1 / S2 / S3 / S4 score-based assignment policies
    metrics        - numTaskFinished, throughput, wait time, load balance

Design contract
---------------
Every scheduler produces the *same* output signature as the official API::

    schedule[i] = task_id   or   -1

and is called at planning updates with a soft ``time_limit``.  Path planning
and execution are identical across schedulers, so any metric difference is
attributable to the assignment policy alone -- which is exactly the
controlled ablation the proposal asks for.

The module depends only on the Python standard library at simulation time so
that it runs anywhere; numpy is used by the analysis layer only.
"""

__version__ = "1.0.0"

from .instance import Instance, load_instance, discover_instances
from .grid import Grid
from .schedulers import SCHEDULERS, make_scheduler
from .world import Simulation, SimResult
from .runner import run_sim, run_from_path

__all__ = [
    "Instance",
    "load_instance",
    "discover_instances",
    "Grid",
    "SCHEDULERS",
    "make_scheduler",
    "Simulation",
    "SimResult",
    "run_sim",
    "run_from_path",
]
