#!/usr/bin/env bash
# Generate a ground-effect training dataset by running many episodes, each
# over a different randomly-seeded (irregular) terrain, and merging the
# per-episode CSV logs into one big CSV.
#
# Columns are exactly what a real drone could sense (motor current x4,
# gyro x3, rangefinder, motor rpm x4, commanded thrust) plus the physics
# engine's ground-truth ground-effect factor per rotor as the training
# label -- see sim/main.c's --csv header for the full column list.
#
# Usage:
#   ./generate_dataset.sh [out_dir] [n_episodes] [episode_duration_s] [start_seed]
#
# Runs with --fast (no real-time pacing), so this is CPU-bound, not
# wall-clock-bound -- a few hundred episodes typically takes well under a
# minute.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

OUT_DIR="${1:-$SCRIPT_DIR/dataset}"
N_EPISODES="${2:-200}"
DURATION="${3:-60}"
START_SEED="${4:-1}"

echo "[generate_dataset] configuring/building"
cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=RelWithDebInfo >/dev/null
make -C "$BUILD_DIR" -j"$(nproc)" >/dev/null

mkdir -p "$OUT_DIR"
MERGED="$OUT_DIR/dataset_all.csv"
TMP_EP="$(mktemp)"
trap 'rm -f "$TMP_EP"' EXIT

rm -f "$MERGED"
t_start=$(date +%s)

for ((i = 0; i < N_EPISODES; i++)); do
    seed=$((START_SEED + i))
    "$BUILD_DIR/bin/simulator" --fc "$BUILD_DIR/bin/libfc.so" --seed "$seed" \
        --duration "$DURATION" --fast --no-telemetry --csv "$TMP_EP" >/dev/null

    if [ ! -s "$MERGED" ]; then
        cp "$TMP_EP" "$MERGED"
    else
        tail -n +2 "$TMP_EP" >> "$MERGED"
    fi

    if (( (i + 1) % 20 == 0 || i + 1 == N_EPISODES )); then
        echo "[generate_dataset] episode $((i + 1))/$N_EPISODES done (seed=$seed)"
    fi
done

t_end=$(date +%s)
n_rows=$(($(wc -l < "$MERGED") - 1))
echo "[generate_dataset] wrote $n_rows rows from $N_EPISODES episodes to $MERGED in $((t_end - t_start))s"
