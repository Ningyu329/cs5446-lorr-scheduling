"""PlanViz-style post-hoc spatial analysis of executed paths (headless).

PlanViz (MAPF-Competition/PlanViz) is an interactive Tk viewer; every run keeps
its start-kit output as ``results/runs/<phase>/<job>/output.json.gz`` so it can
be opened there directly (``gunzip`` first).  For the report we need the same
information as static figures, so this module decodes the compressed
``actualPaths`` strings and accumulates, per grid cell,

* occupancy ticks  -- where robots spend time at all, and
* waiting ticks    -- ticks spent on ``W`` (planned waits and delay stops),

which is exactly what one inspects in PlanViz when looking for bottlenecks.

Path string grammar (start-kit 3.1, ``Input_Output_Format.md``)::

    [(t,row,col,ori,counter):(A n,A n,...)][(...):(...)]...

with ``A`` in ``F`` (forward), ``R``/``C`` (rotate), ``W`` (wait), ``T`` (no
action) and ``n`` the duration in ticks.  Forward and rotate actions take
``agentMaxCounter`` ticks; the snapshot ``counter`` gives the progress of the
action in flight at the segment start.  Robots never stand on obstacles, which
gives a built-in check of the decoder (``obstacle_share`` below).
"""

from __future__ import annotations

import gzip
import json
import os
import re
import sys

import numpy as np

SEG = re.compile(r"\[\((\d+),(\d+),(\d+),(-?\d+),(-?\d+)\):\(([^)]*)\)\]")
DR = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 0:east 1:south 2:west 3:north


def decode(path_str, rows, cols, ctr, occ, wait, t_from=0):
    # Start-kit 3.1.0 writes chunked_actual_snapshot_states[i][0] for *every*
    # segment (Simulator::actual_path_to_json), i.e. all snapshots repeat the
    # initial state; so only the first snapshot is used and the state is carried.
    first = True
    for m in SEG.finditer(path_str):
        if first:
            t, r, c, ori, cnt = (int(x) for x in m.groups()[:5])
            prog = max(0, cnt)
            first = False
        for tok in m.group(6).split(","):
            tok = tok.strip()
            if not tok:
                continue
            a, n = tok.split()
            n = int(n)
            for _ in range(n):
                if t >= t_from:
                    occ[r, c] += 1
                    if a in "WT":
                        wait[r, c] += 1
                t += 1
                if a in "FRC":
                    prog += 1
                    if prog >= ctr:
                        prog = 0
                        if a == "F":
                            dr, dc = DR[ori % 4]
                            if 0 <= r + dr < rows and 0 <= c + dc < cols:
                                r, c = r + dr, c + dc
                        elif a == "R":
                            ori = (ori + 1) % 4
                        else:
                            ori = (ori + 3) % 4
                # W / T pause an action in flight without resetting its progress


def heatmaps(output_json_gz, rows, cols, t_from=0):
    with gzip.open(output_json_gz, "rt") as f:
        out = json.load(f)
    ctr = out.get("agentMaxCounter", 1)
    occ = np.zeros((rows, cols))
    wait = np.zeros((rows, cols))
    for p in out["actualPaths"]:
        decode(p, rows, cols, ctr, occ, wait, t_from)
    return occ, wait, out


def read_map(path):
    with open(path) as f:
        lines = f.read().splitlines()
    h = int(lines[1].split()[1])
    w = int(lines[2].split()[1])
    grid = np.array([[ch in "@T" for ch in row[:w]] for row in lines[4:4 + h]])
    return h, w, grid


if __name__ == "__main__":
    job = sys.argv[1]
    print(job)
