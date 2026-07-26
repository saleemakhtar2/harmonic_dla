"""Deterministic per-walker random streams for compiled kernels."""

from __future__ import annotations

import numpy as np
from numba import njit

_MASK = np.uint64(0xFFFFFFFFFFFFFFFF)
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_SM64_A = np.uint64(0xBF58476D1CE4E5B9)
_SM64_B = np.uint64(0x94D049BB133111EB)
_XOSHIRO_MUL = np.uint64(5)
_XOSHIRO_ROT_MUL = np.uint64(9)
_TWO_NEG_53 = 1.0 / 9007199254740992.0


@njit(cache=True, inline="always")
def _rotl(value: np.uint64, shift: int) -> np.uint64:
    return ((value << np.uint64(shift)) | (value >> np.uint64(64 - shift))) & _MASK


@njit(cache=True, inline="always")
def _splitmix64(value: np.uint64) -> tuple[np.uint64, np.uint64]:
    value = (value + _GOLDEN) & _MASK
    z = value
    z = ((z ^ (z >> np.uint64(30))) * _SM64_A) & _MASK
    z = ((z ^ (z >> np.uint64(27))) * _SM64_B) & _MASK
    z ^= z >> np.uint64(31)
    return value, z & _MASK


@njit(cache=True)
def seed_state(master_seed: int, stream: int) -> np.ndarray:
    """Create one xoshiro256** state from a master seed and stream id."""
    # Addition preserves both inputs through the SplitMix avalanche. XOR would
    # make the two-dimensional seed/stream domain trivially collide.
    mixed = np.uint64(master_seed) + (np.uint64(stream) + np.uint64(1)) * _GOLDEN
    state = np.empty(4, dtype=np.uint64)
    for i in range(4):
        mixed, state[i] = _splitmix64(mixed)
    if (state[0] | state[1] | state[2] | state[3]) == np.uint64(0):
        state[0] = _GOLDEN
    return state


@njit(cache=True, inline="always")
def next_u64(state: np.ndarray) -> np.uint64:
    """Advance xoshiro256** and return one unsigned 64-bit integer."""
    result = _rotl((state[1] * _XOSHIRO_MUL) & _MASK, 7)
    result = (result * _XOSHIRO_ROT_MUL) & _MASK
    t = (state[1] << np.uint64(17)) & _MASK

    state[2] ^= state[0]
    state[3] ^= state[1]
    state[1] ^= state[2]
    state[0] ^= state[3]
    state[2] ^= t
    state[3] = _rotl(state[3], 45)
    return result


@njit(cache=True, inline="always")
def uniform_open(state: np.ndarray) -> float:
    """Return a deterministic float strictly inside ``(0, 1)``."""
    bits = next_u64(state) >> np.uint64(11)
    return (float(bits) + 0.5) * _TWO_NEG_53
