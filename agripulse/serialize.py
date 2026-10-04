"""Model serialization and edge weight compression for AgriPulse.

Ensures that model weights satisfy the hardware constraints of the World Bank Hackathon:
1. Total binary footprint < 80 KB (FP32 ~56 KB, FP16 ~28 KB)
2. Fast zero-overhead deserialization on low-power ARM CPUs
3. Self-contained metadata (normalization statistics, temperature scaling parameter,
   and calibrated decision thresholds).
"""

from dataclasses import asdict
import json
import os
import struct
from typing import Any, Dict, Optional, Tuple
import flax
from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np

from agripulse.dataset import NormalizationStats


MAGIC_HEADER = b"AGRIV1"


def flatten_pytree(tree: Dict[str, Any]) -> Tuple[np.ndarray, list]:
    """Flatten nested parameter pytree into a 1D contiguous numpy array and structural manifest."""
    leaves, treedef = jax.tree_util.tree_flatten_with_path(tree)
    manifest = []
    arrays = []

    for path, leaf in leaves:
        # Path string representation e.g. "params/conv1/kernel"
        path_str = "/".join(
            p.key if hasattr(p, "key") else str(p) for p in path
        )
        arr = np.array(leaf, dtype=np.float32)
        manifest.append({
            "path": path_str,
            "shape": list(arr.shape),
            "size": int(arr.size),
            "dtype": str(arr.dtype),
        })
        arrays.append(arr.reshape(-1))

    flat_weights = np.concatenate(arrays, axis=0) if arrays else np.array([], dtype=np.float32)
    return flat_weights, manifest


def unflatten_pytree(
    flat_weights: np.ndarray,
    manifest: list,
    target_pytree_template: Dict[str, Any],
) -> Dict[str, Any]:
    """Reconstruct nested parameter dictionary from flat weights array using manifest."""
    offset = 0
    reconstructed_leaves = []

    for item in manifest:
        size = item["size"]
        shape = tuple(item["shape"])
        arr_flat = flat_weights[offset : offset + size]
        arr = arr_flat.reshape(shape)
        reconstructed_leaves.append(arr)
        offset += size

    # Use template tree definition to rebuild dictionary
    _, treedef = jax.tree_util.tree_flatten_with_path(target_pytree_template)
    reconstructed = treedef.unflatten(reconstructed_leaves)
    return reconstructed


def export_model_artifacts(
    params: Dict[str, Any],
    batch_stats: Dict[str, Any],
    temperature: float,
    metrics: Dict[str, float],
    norm_stats_path: str = "data/norm_stats.json",
    output_dir: str = "models",
) -> Dict[str, str]:
    """Serialize full model checkpoint, compressed binary edge bundle, and metadata."""
    os.makedirs(output_dir, exist_ok=True)

    # Load normalization stats
    if os.path.exists(norm_stats_path):
        with open(norm_stats_path, "r", encoding="utf-8") as f:
            norm_stats_dict = json.load(f)
    else:
        norm_stats_dict = {}

    # 1. Standard Flax MsgPack Serialization
    full_state = {
        "params": params,
        "batch_stats": batch_stats,
    }
    msgpack_bytes = serialization.to_bytes(full_state)
    msgpack_path = os.path.join(output_dir, "agripulse_model.msgpack")
    with open(msgpack_path, "wb") as f:
        f.write(msgpack_bytes)
    msgpack_size_kb = len(msgpack_bytes) / 1024.0

    # 2. Ultra-compact Edge Binary Export (FP16 / FP32)
    flat_params, param_manifest = flatten_pytree(params)
    flat_bn, bn_manifest = flatten_pytree(batch_stats)

    combined_flat = np.concatenate([flat_params, flat_bn], axis=0).astype(np.float32)
    total_weights_count = len(combined_flat)

    # Write compressed FP32 edge binary
    edge_bin_path = os.path.join(output_dir, "agripulse_edge_fp32.bin")
    with open(edge_bin_path, "wb") as f:
        # Header: MAGIC (6 bytes) + version (uint16) + total_weights (uint32) + temperature (float32)
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<HI", 1, total_weights_count))
        f.write(struct.pack("<f", float(temperature)))
        f.write(combined_flat.tobytes())
    bin_size_kb = os.path.getsize(edge_bin_path) / 1024.0

    # Write half-precision FP16 edge binary (for extreme hardware constraints)
    fp16_bin_path = os.path.join(output_dir, "agripulse_edge_fp16.bin")
    with open(fp16_bin_path, "wb") as f:
        f.write(MAGIC_HEADER)
        f.write(struct.pack("<HI", 2, total_weights_count))  # version 2 = fp16
        f.write(struct.pack("<f", float(temperature)))
        f.write(combined_flat.astype(np.float16).tobytes())
    fp16_size_kb = os.path.getsize(fp16_bin_path) / 1024.0

    # 3. Model Metadata & Specification File
    metadata = {
        "model_name": "AgriPulse",
        "architecture": "1D-Temporal-CNN (3-stage Residual)",
        "input_shape": [1, 30, 6],
        "total_parameters": total_weights_count,
        "calibration": {
            "temperature": float(temperature),
            "method": "Validation Negative Log-Likelihood Minimization",
            "decision_thresholds": {
                "stable_risk_max": 0.40,
                "uncertain_zone": [0.40, 0.75],
                "critical_action_min": 0.75,
            },
            "guardrail_fail_safe": {
                "enabled": True,
                "swahili_message": "Hali ya hewa si ya kawaida. Wasiliana na afisa ugani au mkuu wa ushirika",
                "english_message": "Unusual microclimatic pattern. Consult cooperative extension officer.",
            },
        },
        "test_performance": metrics,
        "binary_sizes": {
            "msgpack_kb": round(msgpack_size_kb, 2),
            "edge_bin_fp32_kb": round(bin_size_kb, 2),
            "edge_bin_fp16_kb": round(fp16_size_kb, 2),
            "hardware_constraint_limit_kb": 80.0,
            "meets_constraint": bool(bin_size_kb < 80.0),
        },
        "normalization": norm_stats_dict,
        "param_manifest": param_manifest,
        "bn_manifest": bn_manifest,
    }

    meta_path = os.path.join(output_dir, "agripulse_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 60)
    print("MODEL SERIALIZATION SUMMARY")
    print("=" * 60)
    print(f"Flax MsgPack:  {msgpack_path} ({msgpack_size_kb:.2f} KB)")
    print(f"Edge FP32 Bin: {edge_bin_path} ({bin_size_kb:.2f} KB)")
    print(f"Edge FP16 Bin: {fp16_bin_path} ({fp16_size_kb:.2f} KB)")
    print(f"Metadata JSON: {meta_path}")
    print(f"Small AI Constraint (< 80 KB): {'PASSED' if bin_size_kb < 80.0 else 'FAILED'}")
    print("=" * 60 + "\n")

    return {
        "msgpack": msgpack_path,
        "edge_fp32": edge_bin_path,
        "edge_fp16": fp16_bin_path,
        "metadata": meta_path,
    }


def load_model_from_artifacts(
    model_dir: str = "models",
) -> Tuple[Dict[str, Any], Dict[str, Any], float, Dict[str, Any]]:
    """Load model parameters, batch_stats, temperature, and metadata for inference."""
    meta_path = os.path.join(model_dir, "agripulse_metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    temperature = float(metadata["calibration"]["temperature"])

    # Load weights from msgpack
    msgpack_path = os.path.join(model_dir, "agripulse_model.msgpack")
    with open(msgpack_path, "rb") as f:
        msgpack_bytes = f.read()

    # Construct target structure template
    from agripulse.model import AgriPulse1DCNN

    model = AgriPulse1DCNN()
    dummy_x = jnp.ones((1, 30, 6), dtype=jnp.float32)
    template_vars = model.init(jax.random.PRNGKey(0), dummy_x, train=False)

    restored_vars = serialization.from_bytes(template_vars, msgpack_bytes)
    params = restored_vars["params"]
    batch_stats = restored_vars["batch_stats"]

    return params, batch_stats, temperature, metadata
