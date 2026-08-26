"""
Shared dataset loading/splitting/windowing for all ground-effect model
training scripts (train_models.py: classical sklearn models;
train_all_models.py: classical + PyTorch models).
"""
from pathlib import Path
import random

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "gyro_x", "gyro_y", "gyro_z",
    "rangefinder_alt",
    "motor_rpm_0", "motor_rpm_1", "motor_rpm_2", "motor_rpm_3",
    "motor_current_0", "motor_current_1", "motor_current_2", "motor_current_3",
    "thrust_cmd_norm",
]
TARGET_COL = "ge_factor_truth_mean"
BASELINE_COL = "ge_ff_analytical"
DEFAULT_GROUP_SPLIT_SEED = 20260825


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL, BASELINE_COL])
    # Sort by (seed, t), not just t: every episode's local clock restarts at
    # 0, so multi-episode datasets have many rows sharing the same t. A
    # plain sort_values("t") interleaves rows from different episodes at
    # matching timestamps, which silently destroys the "contiguous block"
    # assumption block_split()/build_windows()/build_pointwise_xy() rely on
    # -- each episode must stay internally time-ordered *and* grouped
    # together, not globally time-ordered across episodes.
    df = df.sort_values(["seed", "t"]).reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"[data] dropped {dropped} malformed/NaN rows (likely a truncated last line)")
    print(f"[data] loaded {len(df)} rows, {df['seed'].nunique()} episode(s), "
          f"t in [{df['t'].min():.1f}, {df['t'].max():.1f}]s")
    return df


def add_block_id(df: pd.DataFrame, block_s: float = 10.0) -> pd.DataFrame:
    """Tags each row with a block id: a contiguous block_s-second chunk of
    a single episode. Distinct episodes (different seed) never share a
    block id even if their local time t happens to overlap."""
    df = df.copy()
    df["block_id"] = df["seed"].astype(str) + "_" + (df["t"] // block_s).astype(int).astype(str)
    return df


def block_split(df: pd.DataFrame, test_frac: float, block_s: float = 10.0, seed: int = 0):
    """Chop each episode into contiguous block_s-second blocks and
    randomly hold out whole blocks for testing, scattered across the
    whole track/dataset.

    A plain random row split would leak badly (rows are ~60Hz samples of a
    continuous flight, so neighbors are nearly identical). A single
    "last N%" time split avoids that leak but is a coin flip on whether
    that tail happens to be a representative stretch of terrain -- on a
    single long flight it can easily land on an unusually flat/uniform
    segment and give a misleading test score. Holding out many small,
    scattered blocks instead keeps each block internally contiguous (so
    leakage is limited to block edges) while still giving the test set a
    representative mix of terrain conditions. """
    if "block_id" not in df.columns:
        df = add_block_id(df, block_s)
    unique_blocks = df["block_id"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_blocks)
    n_test_blocks = max(1, int(len(unique_blocks) * test_frac))
    test_blocks = set(unique_blocks[:n_test_blocks])
    is_test = df["block_id"].isin(test_blocks)
    train_df, test_df = df[~is_test].copy(), df[is_test].copy()
    print(f"[data] split: {len(train_df)} train rows / {len(test_df)} test rows "
          f"({n_test_blocks}/{len(unique_blocks)} {block_s:.0f}s blocks held out)")
    return train_df, test_df


def seed_group_split(df: pd.DataFrame, train_count=240, val_count=30,
                     test_count=30, random_seed=DEFAULT_GROUP_SPLIT_SEED):
    """Deterministically split whole terrain seeds with zero cross-split leakage."""
    seeds = sorted(int(s) for s in df["seed"].unique())
    expected = train_count + val_count + test_count
    if len(seeds) != expected:
        raise ValueError(f"group split requires exactly {expected} unique seeds, found {len(seeds)}")
    shuffled = seeds.copy()
    random.Random(random_seed).shuffle(shuffled)
    split_seeds = {
        "train": sorted(shuffled[:train_count]),
        "validation": sorted(shuffled[train_count:train_count + val_count]),
        "test": sorted(shuffled[train_count + val_count:]),
    }
    assert_seed_splits_disjoint(split_seeds)
    frames = {
        name: df[df["seed"].isin(values)].copy()
        for name, values in split_seeds.items()
    }
    for name, frame in frames.items():
        actual = set(int(s) for s in frame["seed"].unique())
        if actual != set(split_seeds[name]):
            raise AssertionError(f"{name} row membership does not match recorded seed list")
    return frames["train"], frames["validation"], frames["test"], split_seeds


def assert_seed_splits_disjoint(split_seeds):
    required = {"train", "validation", "test"}
    if set(split_seeds) != required:
        raise ValueError(f"split keys must be {sorted(required)}")
    sets = {name: set(map(int, values)) for name, values in split_seeds.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"seed leakage between {a} and {b}: {sorted(overlap)}")
    if any(not values for values in sets.values()):
        raise ValueError("all seed splits must be non-empty")
    return True


def infer_dt(df: pd.DataFrame) -> float:
    """Median sample interval [s], inferred per-block so it's not thrown
    off by the (large) gap between the last row of one block and the
    first of the next. Used to convert a --lead-s (seconds) argument into
    a row-count shift without hardcoding the telemetry rate."""
    if "block_id" not in df.columns:
        df = add_block_id(df)
    deltas = df.groupby("block_id")["t"].diff()
    deltas = deltas[deltas > 0]
    if len(deltas) == 0:
        return 1.0 / 60.0
    return float(deltas.median())


def lead_s_to_rows(df: pd.DataFrame, lead_s: float) -> int:
    if lead_s <= 0:
        return 0
    dt = infer_dt(df)
    rows = max(1, round(lead_s / dt))
    print(f"[data] inferred dt={dt * 1000:.1f}ms -> lead_s={lead_s:.3f}s is {rows} row(s) ahead")
    return rows


def _iter_blocks(block_ids: np.ndarray):
    """Yields (start, end) index pairs for each maximal contiguous run of
    identical block_id -- still correct after block_split filters some
    blocks out, since that only removes whole chunks, it never reorders
    or interleaves the rows that remain."""
    start = 0
    n = len(block_ids)
    while start < n:
        end = start + 1
        while end < n and block_ids[end] == block_ids[start]:
            end += 1
        yield start, end
        start = end


def build_windows(df: pd.DataFrame, window: int, lead_rows: int = 0,
                   feature_cols=FEATURE_COLS, target_col=TARGET_COL):
    """Sliding windows of `window` consecutive rows *within* a single
    block_id only (never crossing a block boundary, since blocks aren't
    time-contiguous with each other once split).

    lead_rows == 0 : each window's label is the target at its own last
                      (most recent) row -- nowcasting.
    lead_rows  > 0  : each window's label is the target `lead_rows` rows
                      *after* the window's last row -- forecasting, which
                      is what an actual feedforward controller needs: a
                      prediction of where ground effect will be by the
                      time inference + the control loop have caught up,
                      not where it is right now.

    Returns X: [N, window, n_features], y: [N], block_id: [N] (for
    assigning each window to train/test consistently with block_split).
    """
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    block_ids = df["block_id"].to_numpy()

    X_list, y_list, b_list = [], [], []
    for start, end in _iter_blocks(block_ids):
        block_len = end - start
        if block_len < window + lead_rows:
            continue
        block_feats = feats[start:end]
        block_targets = targets[start:end]
        for i in range(window - 1, block_len - lead_rows):
            X_list.append(block_feats[i - window + 1: i + 1])
            y_list.append(block_targets[i + lead_rows])
            b_list.append(block_ids[start])

    X = np.stack(X_list).astype(np.float32) if X_list else np.zeros((0, window, len(feature_cols)), np.float32)
    y = np.array(y_list, dtype=np.float32)
    b = np.array(b_list)
    return X, y, b


def build_pointwise_xy(df: pd.DataFrame, lead_rows: int = 0,
                        feature_cols=FEATURE_COLS, target_col=TARGET_COL, extra_cols=()):
    """Single-timestep equivalent of build_windows: features at row i,
    target at row i + lead_rows (same block only). `extra_cols` (e.g. the
    analytical baseline) are also pulled from row i, so the baseline is
    compared on the exact same forecasting task -- "given what you know
    up to now, what will the target be lead_rows ahead" -- as the trained
    models, instead of getting an easier now-vs-now comparison.
    """
    feats = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[target_col].to_numpy(dtype=np.float32)
    block_ids = df["block_id"].to_numpy()
    extra_arrs = {c: df[c].to_numpy(dtype=np.float32) for c in extra_cols}

    if lead_rows == 0:
        out_extra = {c: extra_arrs[c] for c in extra_cols}
        return feats, targets, block_ids, out_extra

    X_list, y_list, b_list = [], [], []
    extra_lists = {c: [] for c in extra_cols}
    for start, end in _iter_blocks(block_ids):
        block_len = end - start
        if block_len <= lead_rows:
            continue
        for i in range(0, block_len - lead_rows):
            X_list.append(feats[start + i])
            y_list.append(targets[start + i + lead_rows])
            b_list.append(block_ids[start])
            for c in extra_cols:
                extra_lists[c].append(extra_arrs[c][start + i])

    X = np.stack(X_list).astype(np.float32) if X_list else np.zeros((0, len(feature_cols)), np.float32)
    y = np.array(y_list, dtype=np.float32)
    b = np.array(b_list)
    out_extra = {c: np.array(v, dtype=np.float32) for c, v in extra_lists.items()}
    return X, y, b, out_extra
