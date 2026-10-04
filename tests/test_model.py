"""Tests for Flax Linen 1D-CNN Model Architecture."""

import jax
import jax.numpy as jnp
import pytest

from agripulse.model import AgriPulse1DCNN, count_parameters


def test_model_forward_and_shapes():
    model = AgriPulse1DCNN()
    rng = jax.random.PRNGKey(42)

    for batch_size in [1, 4, 16]:
        dummy_x = jnp.zeros((batch_size, 30, 6), dtype=jnp.float32)
        variables = model.init(rng, dummy_x, train=False)
        out = model.apply(variables, dummy_x, train=False)

        assert out.shape == (batch_size,), f"Expected shape ({batch_size},), got {out.shape}"
        assert jnp.all(jnp.isfinite(out)), "Model output contains NaN or Inf"


def test_parameter_count_and_size_constraint():
    model = AgriPulse1DCNN()
    rng = jax.random.PRNGKey(0)
    dummy_x = jnp.zeros((1, 30, 6), dtype=jnp.float32)
    variables = model.init(rng, dummy_x, train=False)

    total_params, size_kb = count_parameters(variables["params"])
    assert total_params < 25000, f"Too many parameters: {total_params} >= 25,000"
    assert size_kb < 80.0, f"Model size exceeds 80 KB: {size_kb:.2f} KB"
    assert total_params > 5000, f"Model suspiciously small: {total_params}"


def test_batch_norm_modes():
    model = AgriPulse1DCNN()
    rng = jax.random.PRNGKey(123)
    dummy_x = jax.random.normal(rng, (8, 30, 6))

    variables = model.init(rng, dummy_x, train=True)
    initial_mean = variables["batch_stats"]["bn1"]["mean"]

    # In train mode, batch_stats should be mutable and update
    _, new_vars = model.apply(variables, dummy_x, train=True, mutable=["batch_stats"])
    updated_mean = new_vars["batch_stats"]["bn1"]["mean"]

    assert not jnp.allclose(initial_mean, updated_mean), "BatchNorm stats did not update in train mode"
