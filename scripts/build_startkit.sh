#!/bin/bash
# Installs the CDA scheduler into the official LoRR start-kit and builds ./build/lifelong.
# Usage: scripts/build_startkit.sh [extra CMake arguments]   (run from anywhere)
set -euo pipefail
PROJ="$(cd "$(dirname "$0")/.." && pwd)"
SK="$PROJ/third_party/Start-Kit"
PIN=9bc75d9dd9883de049294ab2981bf325bbe19d49
PYTHON="${PYTHON:-python3}"
JOBS="${JOBS:-2}"
for tool in git cmake "$PYTHON"; do
    command -v "$tool" >/dev/null || { echo "Required command not found: $tool" >&2; exit 1; }
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo 'JOBS must be a positive integer' >&2; exit 1; }
if [ ! -d "$SK" ]; then
    git clone https://github.com/MAPF-Competition/Start-Kit "$SK"
    git -C "$SK" checkout "$PIN"
fi
if [ "$(git -C "$SK" rev-parse HEAD)" != "$PIN" ]; then
    echo "Start-Kit is not at the required commit $PIN; use a separate checkout or restore that version first." >&2
    exit 1
fi
cp "$PROJ/sched/TaskScheduler.cpp" "$SK/src/TaskScheduler.cpp"
cp "$PROJ/sched/cda_scheduler.cpp" "$SK/src/cda_scheduler.cpp"
cp "$PROJ/sched/cda_scheduler.h" "$SK/inc/cda_scheduler.h"
PYTHON_EXE="$(command -v "$PYTHON")"
CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF
    -DPYBIND11_FINDPYTHON=ON "-DPython_EXECUTABLE=$PYTHON_EXE")
# pip-installed pybind11 lives inside the selected Python environment.
if PYBIND_DIR="$("$PYTHON" -m pybind11 --cmakedir 2>/dev/null)"; then
    CMAKE_ARGS+=("-Dpybind11_DIR=$PYBIND_DIR")
fi
cmake -B "$SK/build" -S "$SK" "${CMAKE_ARGS[@]}" "$@" > "$SK/cmake.log" 2>&1 || { cat "$SK/cmake.log"; exit 1; }
cmake --build "$SK/build" --target lifelong --parallel "$JOBS" > "$SK/build.log" 2>&1 || { cat "$SK/build.log"; exit 1; }
echo "Built $SK/build/lifelong (Start-Kit ${PIN:0:7})"
