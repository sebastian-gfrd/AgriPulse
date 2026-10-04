"""Tests for Zero-Dependency MicroRuntime and mathematical parity with JAX."""

import numpy as np
import pytest

from agripulse.engine import AgriPulseEdgeEngine
from agripulse.micro_runtime import AgriPulseMicroEngine


def test_micro_runtime_parity():
    jax_engine = AgriPulseEdgeEngine()
    micro_engine = AgriPulseMicroEngine()

    data = np.load("data/ondera_coffee_rust_dataset.npz")
    X_test_raw = data["X_test_raw"]

    diffs = []
    for i in range(15):
        sample = X_test_raw[i]
        jax_prob = jax_engine.predict(sample).risk_score
        micro_prob = micro_engine.predict(sample)
        diff = abs(jax_prob - micro_prob)
        diffs.append(diff)

    max_diff = max(diffs)
    assert max_diff < 1e-4, f"MicroEngine and JAX Engine diverged: max diff = {max_diff}"
