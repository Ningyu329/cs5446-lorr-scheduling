#!/bin/bash
# Installs the CDA scheduler into the official LoRR start-kit and builds ./build/lifelong.
# Usage: scripts/build_startkit.sh   (run from anywhere)
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
SK="$PROJ/third_party/Start-Kit"
if [ ! -d "$SK" ]; then
    git clone https://github.com/MAPF-Competition/Start-Kit "$SK"
    git -C "$SK" checkout 9bc75d9dd9883de049294ab2981bf325bbe19d49
fi
cp "$PROJ/sched/TaskScheduler.cpp" "$SK/src/TaskScheduler.cpp"
cp "$PROJ/sched/cda_scheduler.cpp" "$SK/src/cda_scheduler.cpp"
cp "$PROJ/sched/cda_scheduler.h" "$SK/inc/cda_scheduler.h"
cmake -B "$SK/build" "$SK" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF > "$SK/cmake.log" 2>&1 || { cat "$SK/cmake.log"; exit 1; }
make -C "$SK/build" -j 32 lifelong 2>&1 | grep -E "error|warning: unused|Built target" || true
ls -la "$SK/build/lifelong"
