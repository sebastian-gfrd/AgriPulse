"""1D Temporal Convolutional Neural Network for microclimatic incubation detection.

Designed for Coffee Leaf Rust (Hemileia vastatrix) anticipation 15-25 days
prior to visual symptom eruption.
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn


class AgriPulse1DCNN(nn.Module):
    """Ultra-lightweight 1D Temporal CNN for Edge Microclimate Analysis.

    Evaluates a 30-day sliding window of 6 environmental channels to detect
    favorable microclimatic conditions for coffee rust germination and incubation.
    Total parameter footprint is < 25,000 parameters (< 80 KB).
    """

    num_classes: int = 1

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch, 30, 6)
            train: Whether the model is in training mode (updates BatchNorm stats)

        Returns:
            Logits of shape (batch,)
        """
        # Stage 1: Short-term microclimatic pattern extraction (3-5 day windows)
        h = nn.Conv(features=16, kernel_size=(5,), padding="SAME", name="conv1")(x)
        h = nn.BatchNorm(use_running_average=not train, name="bn1")(h)
        h = nn.gelu(h)
        h = nn.max_pool(h, window_shape=(2,), strides=(2,))  # Shape: (batch, 15, 16)

        # Stage 2: Residual block for medium-term cumulative dynamics (7-14 days)
        res = nn.Conv(features=32, kernel_size=(1,), padding="SAME", name="conv_res_proj")(h)
        h_res = nn.Conv(features=32, kernel_size=(3,), padding="SAME", name="conv2_1")(h)
        h_res = nn.BatchNorm(use_running_average=not train, name="bn2_1")(h_res)
        h_res = nn.gelu(h_res)
        h_res = nn.Conv(features=32, kernel_size=(3,), padding="SAME", name="conv2_2")(h_res)
        h_res = nn.BatchNorm(use_running_average=not train, name="bn2_2")(h_res)
        h = nn.gelu(h_res + res)
        h = nn.max_pool(h, window_shape=(2,), strides=(2,))  # Shape: (batch, 7, 32)

        # Stage 3: Long-term biological incubation integration (15-30 days)
        h = nn.Conv(features=64, kernel_size=(3,), padding="SAME", name="conv3")(h)
        h = nn.BatchNorm(use_running_average=not train, name="bn3")(h)
        h = nn.gelu(h)  # Shape: (batch, 7, 64)

        # Temporal collapse via Global Average Pooling
        h = jnp.mean(h, axis=1)  # Shape: (batch, 64)

        # Compact Decision Head
        h = nn.Dense(features=32, name="dense1")(h)
        h = nn.gelu(h)
        logits = nn.Dense(features=self.num_classes, name="dense_out")(h)
        return logits.squeeze(-1)


def count_parameters(params) -> Tuple[int, float]:
    """Calculate parameter count and memory size in KB (float32)."""
    total_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    size_kb = (total_params * 4) / 1024.0  # 4 bytes per float32
    return total_params, size_kb
