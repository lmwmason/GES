#!/usr/bin/env bash
# Build (if needed) and run the drone/ground-effect simulator.
#
# Usage:
#   ./run.sh                          build + run simulator only
#   ./run.sh --viz                    build + run simulator and launch the Python 3D viewer
#   ./run.sh --viz-only                launch only the Python 3D viewer (simulator run separately)
#   ./run.sh -- <simulator args...>    forward args to the simulator binary, e.g.:
#                                       ./run.sh --viz -- --seed 7 --duration 40
#                                       (see: build/bin/simulator --help)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

WITH_VIZ=0
VIZ_ONLY=0
SIM_ARGS=()
VIZ_SEED=1

for arg in "$@"; do
    case "$arg" in
        --viz) WITH_VIZ=1 ;;
        --viz-only) VIZ_ONLY=1 ;;
        --) ;;
        *) SIM_ARGS+=("$arg") ;;
    esac
done

# Keep the viewer's terrain in sync with whatever --seed the simulator got.
for i in "${!SIM_ARGS[@]}"; do
    if [ "${SIM_ARGS[$i]}" = "--seed" ] && [ $((i + 1)) -lt ${#SIM_ARGS[@]} ]; then
        VIZ_SEED="${SIM_ARGS[$((i + 1))]}"
    fi
done

run_viz() {
    echo "[run.sh] launching viewer (viz/visualize.py, seed=$VIZ_SEED)"
    python3 "$SCRIPT_DIR/viz/visualize.py" --seed "$VIZ_SEED" &
    echo $! > "$BUILD_DIR/.viz.pid"
}

if [ "$VIZ_ONLY" -eq 1 ]; then
    mkdir -p "$BUILD_DIR"
    run_viz
    wait
    exit 0
fi

echo "[run.sh] configuring/building (cmake + make)"
mkdir -p "$BUILD_DIR"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
make -C "$BUILD_DIR" -j"$(nproc)"

if [ "$WITH_VIZ" -eq 1 ]; then
    run_viz
    trap '[ -f "$BUILD_DIR/.viz.pid" ] && kill "$(cat "$BUILD_DIR/.viz.pid")" 2>/dev/null; rm -f "$BUILD_DIR/.viz.pid"' EXIT
    sleep 1
fi

echo "[run.sh] running simulator"
"$BUILD_DIR/bin/simulator" "${SIM_ARGS[@]}"
