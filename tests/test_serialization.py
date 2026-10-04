"""Tests for Model Serialization, Edge Binaries, and Weight Restoration."""

import json
import os
import numpy as np
import pytest

from agripulse.serialize import (
    export_model_artifacts,
    load_model_from_artifacts,
    flatten_pytree,
    unflatten_pytree,
)
from agripulse.model import AgriPulse1DCNN
import jax
import jax.numpy as jnp


def test_serialization_and_binary_constraints(tmp_path):
    out_dir = str(tmp_path / "test_models")
    model = AgriPulse1DCNN()
    rng = jax.random.PRNGKey(7)
    dummy_x = jnp.zeros((1, 30, 6), dtype=jnp.float32)
    vars = model.init(rng, dummy_x, train=False)

    test_metrics = {"accuracy": 0.85, "f1": 0.75, "auc": 0.90}
    temp = 1.15

    artifacts = export_model_artifacts(
        params=vars["params"],
        batch_stats=vars["batch_stats"],
        temperature=temp,
        metrics=test_metrics,
        norm_stats_path="data/norm_stats.json",
        output_dir=out_dir,
    )

    # Verify on-disk file sizes satisfy World Bank Small AI constraint (< 80 KB)
    assert os.path.exists(artifacts["msgpack"])
    assert os.path.exists(artifacts["edge_fp32"])
    assert os.path.exists(artifacts["edge_fp16"])
    assert os.path.exists(artifacts["metadata"])

    fp32_size_kb = os.path.getsize(artifacts["edge_fp32"]) / 1024.0
    fp16_size_kb = os.path.getsize(artifacts["edge_fp16"]) / 1024.0
    msgpack_size_kb = os.path.getsize(artifacts["msgpack"]) / 1024.0

    assert fp32_size_kb < 80.0, f"FP32 binary too large: {fp32_size_kb:.2f} KB"
    assert fp16_size_kb < 40.0, f"FP16 binary too large: {fp16_size_kb:.2f} KB"
    assert msgpack_size_kb < 80.0, f"Msgpack too large: {msgpack_size_kb:.2f} KB"

    # Verify weight restoration and prediction equivalence
    rest_params, rest_stats, rest_temp, rest_meta = load_model_from_artifacts(out_dir)
    assert abs(rest_temp - temp) < 1e-4

    test_x = jax.random.normal(rng, (2, 30, 6))
    orig_out = model.apply(vars, test_x, train=False)
    rest_out = model.apply({"params": rest_params, "batch_stats": rest_stats}, test_x, train=False)

    assert jnp.allclose(orig_out, rest_out, atol=1e-5), "Restored weights produced divergent outputs!"
