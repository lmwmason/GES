"""
Python re-implementation of physics/terrain.c's seeded terrain generator.

Kept bit-for-bit in sync with the C xorshift32 PRNG and the exact order of
random draws in terrain_init()/feature generation, so viz/visualize.py can
regenerate the identical terrain for a given --seed without any file
exchange with the C simulator. If terrain.c's generator changes, mirror
the change here (same draw order matters, not just the same ranges).
"""
import math

import numpy as np

BUMP, DIP, RIPPLE = 0, 1, 2


class Xorshift32:
    def __init__(self, seed: int):
        state = seed if seed else 0x9E3779B9
        self.state = state & 0xFFFFFFFF
        for _ in range(4):
            self.next_u32()

    def next_u32(self) -> int:
        x = self.state
        x = (x ^ (x << 13)) & 0xFFFFFFFF
        x = (x ^ (x >> 17)) & 0xFFFFFFFF
        x = (x ^ (x << 5)) & 0xFFFFFFFF
        self.state = x
        return x

    def rand01(self) -> float:
        return self.next_u32() / 4294967296.0

    def rand_range(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.rand01()


def generate_features(seed: int):
    rng = Xorshift32(seed)
    count = 800 + int(rng.rand01() * 700.0)  # 800..1499, mirrors terrain.c
    count = min(count, 1500)

    features = []
    x_cursor = 4.0
    for _ in range(count):
        type_roll = rng.rand01()
        ftype = BUMP if type_roll < 0.4 else DIP if type_roll < 0.8 else RIPPLE

        length = rng.rand_range(3.0, 8.0)
        x0 = x_cursor + rng.rand_range(0.5, 3.0)
        x_cursor = x0 + length

        amplitude = rng.rand_range(0.05, 0.22)
        y0 = rng.rand_range(-1.4, 1.4)
        y_sigma = rng.rand_range(0.9, 2.2)
        wavelength = rng.rand_range(1.2, 3.5)
        phase = rng.rand_range(0.0, 2.0 * math.pi)

        features.append((ftype, x0, length, amplitude, y0, y_sigma, wavelength, phase))
    return features


def _feature_height_grid(f, X, Y):
    ftype, x0, length, amplitude, y0, y_sigma, wavelength, phase = f
    mask = (X >= x0) & (X <= x0 + length)
    u = np.clip((X - x0) / length, 0.0, 1.0)
    envelope = 0.5 * (1.0 - np.cos(2.0 * np.pi * u))
    dy = Y - y0
    radial = np.exp(-(dy * dy) / (2.0 * y_sigma * y_sigma))

    if ftype == BUMP:
        h = amplitude * envelope
    elif ftype == DIP:
        h = -amplitude * envelope
    else:
        h = amplitude * envelope * np.sin(2.0 * np.pi * (X - x0) / wavelength + phase)

    roughness = 1.0 + 0.35 * np.sin(5.3 * (X - x0) + phase * 1.7) * envelope
    return np.where(mask, h * radial * roughness, 0.0)


def terrain_height_grid(X, Y, seed: int = 1):
    """Vectorized height field over numpy meshgrid arrays X, Y."""
    features = generate_features(seed)
    H = np.zeros_like(X, dtype=float)
    for f in features:
        H += _feature_height_grid(f, X, Y)
    return H


def terrain_height(x: float, y: float, seed: int = 1) -> float:
    """Scalar height at a single (x, y). Slow (regenerates features each
    call) -- only meant for occasional lookups, not per-frame use."""
    X = np.array([[x]])
    Y = np.array([[y]])
    return float(terrain_height_grid(X, Y, seed)[0, 0])
