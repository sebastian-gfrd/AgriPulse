"""Tests for Pure CPU Edge Engine, Latency, and Guardrail Decisions."""

import os
import numpy as np
import pytest

# Ensure CPU mode
os.environ["JAX_PLATFORMS"] = "cpu"

from agripulse.engine import AgriPulseEdgeEngine


@pytest.fixture(scope="module")
def engine():
    return AgriPulseEdgeEngine(model_dir="models", data_dir="data")


def test_engine_inference_latency_and_ranges(engine):
    raw_window = np.random.uniform(10.0, 30.0, size=(30, 6)).astype(np.float32)
    # Warmup pass
    _ = engine.predict(raw_window)
    # Measured warm run
    res = engine.predict(raw_window)

    assert 0.0 <= res.risk_score <= 1.0, f"Risk score out of [0, 1]: {res.risk_score}"
    assert res.status_code in [0, 1, 2], f"Invalid status code: {res.status_code}"
    assert res.inference_latency_ms < 5.0, f"CPU latency too high: {res.inference_latency_ms:.3f} ms"
    assert len(res.action_swahili) > 10
    assert len(res.action_english) > 10


def test_guardrail_ethical_escalation_branching(engine):
    # Test boundary logic:
    # Status 0: <= 0.40 (Stable, no escalation)
    # Status 1: 0.40 < P < 0.75 (Uncertain, human escalation required)
    # Status 2: >= 0.75 (Critical action)

    data = np.load("data/ondera_coffee_rust_dataset.npz")
    X_test_raw = data["X_test_raw"]

    escalation_tested = False
    stable_tested = False
    critical_tested = False

    for sample in X_test_raw[:50]:
        res = engine.predict(sample)
        if res.status_code == 1:
            assert res.requires_human_escalation is True
            assert "ugani" in res.action_swahili.lower()
            escalation_tested = True
        elif res.status_code == 0:
            assert res.requires_human_escalation is False
            assert "salama" in res.action_swahili.lower()
            stable_tested = True
        elif res.status_code == 2:
            assert res.requires_human_escalation is False
            assert "kutu" in res.action_swahili.lower()
            critical_tested = True

    assert escalation_tested, "Guardrail escalation branch was not triggered across test samples"
    assert stable_tested, "Stable branch was not triggered across test samples"
