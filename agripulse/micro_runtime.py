"""Zero-Dependency Micro-Runtime for AgriPulse.

Executes the AgriPulse 1D-CNN using pure NumPy operations without importing JAX or Flax.
Enables execution on ultra-austere hardware: budget Android runtimes, Raspberry Pi Pico,
MicroPython, or standalone C bindings, proving true Small AI fidelity.
"""

import json
import math
import os
import struct
from typing import Any, Dict, Tuple
import numpy as np

from agripulse.serialize import MAGIC_HEADER


def gelu_exact(x: np.ndarray) -> np.ndarray:
    """Exact GELU activation function using hyperbolic tangent approximation."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def conv1d_same(x: np.ndarray, kernel: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """1D Convolution with SAME padding.

    Args:
        x: Input array of shape (batch, time_steps, in_channels)
        kernel: Weight array of shape (kernel_size, in_channels, out_channels)
        bias: Bias array of shape (out_channels,)
    """
    b, t, c_in = x.shape
    k_len, _, c_out = kernel.shape

    pad_total = k_len - 1
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left

    # Pad temporal dimension
    x_padded = np.pad(x, ((0, 0), (pad_left, pad_right), (0, 0)), mode="constant")

    # Convolution via sliding window dot-product
    out = np.zeros((b, t, c_out), dtype=np.float32)
    for i in range(t):
        window = x_padded[:, i : i + k_len, :]  # (b, k_len, c_in)
        # Tensor contraction over (k_len, c_in)
        out[:, i, :] = np.tensordot(window, kernel, axes=([1, 2], [0, 1])) + bias

    return out


def batch_norm_eval(
    x: np.ndarray,
    scale: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    epsilon: float = 1e-5,
) -> np.ndarray:
    """BatchNorm inference mode execution."""
    inv_std = 1.0 / np.sqrt(var + epsilon)
    return (x - mean) * inv_std * scale + bias


def max_pool1d(x: np.ndarray, window_shape: int = 2, stride: int = 2) -> np.ndarray:
    """1D Max Pooling along temporal axis."""
    b, t, c = x.shape
    t_out = (t - window_shape) // stride + 1
    out = np.zeros((b, t_out, c), dtype=np.float32)
    for i in range(t_out):
        start = i * stride
        end = start + window_shape
        out[:, i, :] = np.max(x[:, start:end, :], axis=1)
    return out


class AgriPulseMicroEngine:
    """Pure NumPy Edge Inference Engine for low-cost embedded and micro devices."""

    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        meta_path = os.path.join(model_dir, "agripulse_metadata.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta = json.load(f)

        self.temperature = float(self.meta["calibration"]["temperature"])

        # Load normalization stats
        norm_path = os.path.join(data_dir, "norm_stats.json")
        with open(norm_path, "r", encoding="utf-8") as f:
            norm_dict = json.load(f)
        self.means = np.array(norm_dict["means"], dtype=np.float32).reshape(1, 1, 6)
        self.stds = np.array(norm_dict["stds"], dtype=np.float32).reshape(1, 1, 6)

        # Unpack binary weights
        self._load_binary_weights(model_dir)

    def _load_binary_weights(self, model_dir: str):
        """Parse raw float32 binary bundle into layer dictionaries."""
        bin_path = os.path.join(model_dir, "agripulse_edge_fp32.bin")
        with open(bin_path, "rb") as f:
            header = f.read(6)
            if header != MAGIC_HEADER:
                raise ValueError(f"Invalid header: {header}")
            version, total_count = struct.unpack("<HI", f.read(6))
            temp_in_hdr = struct.unpack("<f", f.read(4))[0]
            raw_floats = np.frombuffer(f.read(), dtype=np.float32)

        # Reconstruct layer weights from manifest
        param_manifest = self.meta["param_manifest"]
        bn_manifest = self.meta["bn_manifest"]

        self.weights = {}
        offset = 0

        for item in param_manifest:
            size = item["size"]
            shape = tuple(item["shape"])
            arr = raw_floats[offset : offset + size].reshape(shape)
            clean_path = item["path"].replace("params/", "")
            self.weights[clean_path] = arr
            offset += size

        self.bn_stats = {}
        for item in bn_manifest:
            size = item["size"]
            shape = tuple(item["shape"])
            arr = raw_floats[offset : offset + size].reshape(shape)
            clean_path = item["path"].replace("batch_stats/", "")
            self.bn_stats[clean_path] = arr
            offset += size

    def forward(self, x_norm: np.ndarray) -> Tuple[np.ndarray, float]:
        """Execute forward pass using pure vectorized NumPy."""
        if x_norm.ndim == 2:
            x_norm = np.expand_dims(x_norm, axis=0)

        # Stage 1: Short-term microclimatic pattern extraction
        h = conv1d_same(
            x_norm,
            self.weights["conv1/kernel"],
            self.weights["conv1/bias"],
        )
        h = batch_norm_eval(
            h,
            scale=self.weights["bn1/scale"],
            bias=self.weights["bn1/bias"],
            mean=self.bn_stats["bn1/mean"],
            var=self.bn_stats["bn1/var"],
        )
        h = gelu_exact(h)
        h = max_pool1d(h, window_shape=2, stride=2)  # (1, 15, 16)

        # Stage 2: Residual Block
        res = conv1d_same(
            h,
            self.weights["conv_res_proj/kernel"],
            self.weights["conv_res_proj/bias"],
        )
        h_res = conv1d_same(
            h,
            self.weights["conv2_1/kernel"],
            self.weights["conv2_1/bias"],
        )
        h_res = batch_norm_eval(
            h_res,
            scale=self.weights["bn2_1/scale"],
            bias=self.weights["bn2_1/bias"],
            mean=self.bn_stats["bn2_1/mean"],
            var=self.bn_stats["bn2_1/var"],
        )
        h_res = gelu_exact(h_res)

        h_res = conv1d_same(
            h_res,
            self.weights["conv2_2/kernel"],
            self.weights["conv2_2/bias"],
        )
        h_res = batch_norm_eval(
            h_res,
            scale=self.weights["bn2_2/scale"],
            bias=self.weights["bn2_2/bias"],
            mean=self.bn_stats["bn2_2/mean"],
            var=self.bn_stats["bn2_2/var"],
        )
        h = gelu_exact(h_res + res)
        h = max_pool1d(h, window_shape=2, stride=2)  # (1, 7, 32)

        # Stage 3: Long-term biological incubation integration
        h = conv1d_same(
            h,
            self.weights["conv3/kernel"],
            self.weights["conv3/bias"],
        )
        h = batch_norm_eval(
            h,
            scale=self.weights["bn3/scale"],
            bias=self.weights["bn3/bias"],
            mean=self.bn_stats["bn3/mean"],
            var=self.bn_stats["bn3/var"],
        )
        h = gelu_exact(h)  # (1, 7, 64)

        # Global Average Pooling
        gap = np.mean(h, axis=1)  # (1, 64)

        # Decision Head
        d1 = gap @ self.weights["dense1/kernel"] + self.weights["dense1/bias"]
        d1 = gelu_exact(d1)
        logits = d1 @ self.weights["dense_out/kernel"] + self.weights["dense_out/bias"]
        logit_scalar = float(logits.squeeze())

        calibrated_prob = 1.0 / (1.0 + math.exp(-logit_scalar / self.temperature))
        return logits, calibrated_prob

    def predict(self, raw_window: np.ndarray) -> float:
        """Normalize raw input window and return calibrated probability."""
        if raw_window.ndim == 2:
            raw_window = np.expand_dims(raw_window, axis=0)
        x_norm = (raw_window - self.means) / self.stds
        _, prob = self.forward(x_norm)
        return prob
