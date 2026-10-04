"""Tests for Dataset generation, normalization, and physical validity."""

import numpy as np
import pytest

from agripulse.dataset import (
    build_real_empirical_dataset,
    compute_normalization,
    apply_normalization,
    NormalizationStats,
)


def test_real_dataset_shapes_and_physics():
    X, y, ids = build_real_empirical_dataset(seq_length=30, stride=5, seed=99)

    assert len(X) > 100, f"Expected >100 samples, got {len(X)}"
    assert X.shape[1:] == (30, 6)
    assert len(y) == len(X)
    assert len(ids) == len(X)
    assert set(np.unique(y)).issubset({0.0, 1.0})

    # Physical checks
    precip = X[:, :, 0]
    tmax = X[:, :, 1]
    tmin = X[:, :, 2]
    delta_t = X[:, :, 3]
    rh = X[:, :, 4]
    soil = X[:, :, 5]

    assert np.all(precip >= 0.0), "Precipitation must be non-negative"
    assert np.all(tmax >= tmin), "Tmax must be >= Tmin"
    assert np.allclose(delta_t, tmax - tmin, atol=1e-5), "Delta T must match Tmax - Tmin"
    assert np.all((rh >= 20.0) & (rh <= 100.0)), "Relative humidity out of physical bounds"
    assert np.all((soil >= 0.0) & (soil <= 1.0)), "Soil metric out of bounds"


def test_normalization_roundtrip():
    rng = np.random.default_rng(42)
    fake_data = rng.normal(loc=15.0, scale=4.0, size=(50, 30, 6)).astype(np.float32)

    stats = compute_normalization(fake_data)
    normed = apply_normalization(fake_data, stats)

    # Check that normalized data has ~0 mean and ~1 std
    channel_means = np.mean(normed, axis=(0, 1))
    channel_stds = np.std(normed, axis=(0, 1))

    assert np.allclose(channel_means, 0.0, atol=1e-5)
    assert np.allclose(channel_stds, 1.0, atol=1e-5)
