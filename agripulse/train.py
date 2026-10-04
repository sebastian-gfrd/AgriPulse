"""High-performance training and temperature calibration pipeline for AgriPulse.

Utilizes GPU acceleration (NVIDIA RTX 5070 with JAX + CUDA) for fast convergence,
followed by validation temperature scaling for calibrated uncertainty estimation.
Exports the trained model weights and calibration metadata.
"""

import json
import os
import time
from typing import Any, Dict, Tuple
import numpy as np

# Configure JAX memory allocation to avoid preallocation exhaustion on desktop GPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import flax.linen as nn
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from scipy.optimize import minimize_scalar

from agripulse.model import AgriPulse1DCNN, count_parameters
from agripulse.serialize import export_model_artifacts


class TrainStateWithBN(train_state.TrainState):
    """Flax TrainState carrying BatchNorm running statistics."""

    batch_stats: Any


def create_train_state(
    model: nn.Module,
    rng: jax.random.PRNGKey,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    total_steps: int = 1000,
) -> TrainStateWithBN:
    """Initialize model parameters, batch_stats, and AdamW optimizer with cosine schedule."""
    dummy_x = jnp.ones((1, 30, 6), dtype=jnp.float32)
    variables = model.init(rng, dummy_x, train=False)
    params = variables["params"]
    batch_stats = variables["batch_stats"]

    # Cosine decay schedule with linear warmup
    warmup_steps = int(total_steps * 0.1)
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-5,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay),
    )

    return TrainStateWithBN.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        batch_stats=batch_stats,
    )


@jax.jit
def train_step(
    state: TrainStateWithBN,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
    pos_weight: float = 3.5,
) -> Tuple[TrainStateWithBN, jnp.ndarray, jnp.ndarray]:
    """Single JIT-compiled training step on GPU."""

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            batch_x,
            train=True,
            mutable=["batch_stats"],
        )
        # Weighted binary cross-entropy with logits
        bce = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_y)
        weights = jnp.where(batch_y > 0.5, pos_weight, 1.0)
        weighted_loss = jnp.mean(bce * weights)
        return weighted_loss, (logits, updates)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates["batch_stats"])
    return state, loss, logits


@jax.jit
def eval_step(
    state: TrainStateWithBN,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single evaluation step (inference mode, using running batch stats)."""
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": state.batch_stats},
        batch_x,
        train=False,
    )
    bce = optax.sigmoid_binary_cross_entropy(logits=logits, labels=batch_y)
    loss = jnp.mean(bce)
    return loss, logits


def compute_binary_metrics(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute classification metrics: accuracy, precision, recall, F1, ROC-AUC."""
    preds = (probs >= threshold).astype(np.float32)

    tp = float(np.sum((preds == 1.0) & (targets == 1.0)))
    fp = float(np.sum((preds == 1.0) & (targets == 0.0)))
    fn = float(np.sum((preds == 0.0) & (targets == 1.0)))
    tn = float(np.sum((preds == 0.0) & (targets == 0.0)))

    accuracy = (tp + tn) / max(len(targets), 1)
    precision = tp / max(tp + fp, 1e-7)
    recall = tp / max(tp + fn, 1e-7)
    f1 = 2 * precision * recall / max(precision + recall, 1e-7)

    # Approximate ROC-AUC via rank-order
    sorted_indices = np.argsort(probs)
    sorted_targets = targets[sorted_indices]
    n_pos = np.sum(targets == 1.0)
    n_neg = np.sum(targets == 0.0)
    if n_pos > 0 and n_neg > 0:
        ranks = np.arange(1, len(targets) + 1)
        rank_sum_pos = np.sum(ranks[sorted_targets == 1.0])
        auc = (rank_sum_pos - (n_pos * (n_pos + 1) / 2)) / (n_pos * n_neg)
    else:
        auc = 0.5

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def calibrate_temperature(val_logits: np.ndarray, val_targets: np.ndarray) -> float:
    """Optimize Temperature parameter T to calibrate post-sigmoid probabilities.

    Finds T > 0 that minimizes the Negative Log Likelihood (NLL) on the validation set.
    """

    def nll_obj(t: float) -> float:
        scaled_logits = val_logits / t
        # Stable sigmoid NLL
        probs = 1.0 / (1.0 + np.exp(-np.clip(scaled_logits, -30, 30)))
        eps = 1e-7
        probs = np.clip(probs, eps, 1.0 - eps)
        nll = -np.mean(val_targets * np.log(probs) + (1.0 - val_targets) * np.log(1.0 - probs))
        return float(nll)

    res = minimize_scalar(nll_obj, bounds=(0.5, 4.0), method="bounded")
    best_t = float(res.x)
    print(f"Temperature Calibration Complete: Optimal T = {best_t:.4f}")
    return best_t


def compute_expected_calibration_error(
    probs: np.ndarray,
    targets: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(probs)

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper)
        bin_size = np.sum(in_bin)

        if bin_size > 0:
            avg_confidence = np.mean(probs[in_bin])
            avg_accuracy = np.mean(targets[in_bin])
            ece += (bin_size / total_samples) * np.abs(avg_accuracy - avg_confidence)

    return float(ece)


def train_agripulse(
    data_path: str = "data/ondera_coffee_rust_dataset.npz",
    epochs: int = 35,
    batch_size: int = 128,
    learning_rate: float = 2e-3,
    output_dir: str = "models",
    seed: int = 42,
) -> Dict[str, Any]:
    """Full training pipeline with GPU acceleration and validation calibration."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Using JAX Devices: {jax.devices()}")

    # 1. Load Data
    data = np.load(data_path)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    n_train = len(X_train)
    steps_per_epoch = int(np.ceil(n_train / batch_size))
    total_steps = steps_per_epoch * epochs

    # Class balance calculation
    pos_count = np.sum(y_train == 1.0)
    neg_count = np.sum(y_train == 0.0)
    pos_weight = float(neg_count / max(pos_count, 1))
    print(f"Dataset stats: {n_train} train samples ({pos_count} pos, {neg_count} neg, pos_weight={pos_weight:.2f})")

    # 2. Initialize Model & State
    model = AgriPulse1DCNN()
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(
        model,
        init_rng,
        learning_rate=learning_rate,
        total_steps=total_steps,
    )

    total_params, size_kb = count_parameters(state.params)
    print(f"AgriPulse Model initialized: {total_params:,} parameters ({size_kb:.2f} KB FP32)")

    # 3. Training Loop
    start_time = time.time()
    best_val_f1 = 0.0
    best_state = state

    for epoch in range(1, epochs + 1):
        rng, shuffle_rng = jax.random.split(rng)
        perm = np.random.default_rng(int(shuffle_rng[0])).permutation(n_train)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        train_losses = []
        for step in range(steps_per_epoch):
            start_idx = step * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            bx = jnp.array(X_train_shuffled[start_idx:end_idx])
            by = jnp.array(y_train_shuffled[start_idx:end_idx])

            state, loss, _ = train_step(state, bx, by, pos_weight=pos_weight)
            train_losses.append(float(loss))

        # Validation Step
        val_loss, val_logits = eval_step(state, jnp.array(X_val), jnp.array(y_val))
        val_probs = 1.0 / (1.0 + np.exp(-np.array(val_logits)))
        val_metrics = compute_binary_metrics(val_probs, y_val)

        if epoch % 5 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:02d}/{epochs:02d} | "
                f"Train Loss: {np.mean(train_losses):.4f} | "
                f"Val Loss: {float(val_loss):.4f} | "
                f"Val F1: {val_metrics['f1']:.4f} | "
                f"Val Recall: {val_metrics['recall']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            best_state = state

    train_duration = time.time() - start_time
    print(f"Training completed in {train_duration:.2f} seconds ({train_duration/epochs:.3f} s/epoch)")

    # 4. Temperature Scaling Calibration on Validation Set
    _, best_val_logits = eval_step(best_state, jnp.array(X_val), jnp.array(y_val))
    best_val_logits_np = np.array(best_val_logits)
    calibrated_temp = calibrate_temperature(best_val_logits_np, y_val)

    # 5. Final Evaluation on Test Set
    _, test_logits = eval_step(best_state, jnp.array(X_test), jnp.array(y_test))
    test_logits_np = np.array(test_logits)

    uncal_test_probs = 1.0 / (1.0 + np.exp(-test_logits_np))
    cal_test_probs = 1.0 / (1.0 + np.exp(-test_logits_np / calibrated_temp))

    test_metrics = compute_binary_metrics(cal_test_probs, y_test)
    uncal_ece = compute_expected_calibration_error(uncal_test_probs, y_test)
    cal_ece = compute_expected_calibration_error(cal_test_probs, y_test)

    print("\n" + "=" * 60)
    print("FINAL TEST SET BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Accuracy:  {test_metrics['accuracy']*100:.2f}%")
    print(f"Precision: {test_metrics['precision']*100:.2f}%")
    print(f"Recall:    {test_metrics['recall']*100:.2f}%")
    print(f"F1 Score:  {test_metrics['f1']:.4f}")
    print(f"ROC-AUC:   {test_metrics['auc']:.4f}")
    print(f"ECE (Uncalibrated): {uncal_ece:.4f} -> ECE (Calibrated): {cal_ece:.4f}")
    print("=" * 60 + "\n")

    # 6. Export serialized model artifacts for Edge CPU deployment
    artifact_paths = export_model_artifacts(
        params=best_state.params,
        batch_stats=best_state.batch_stats,
        temperature=calibrated_temp,
        metrics=test_metrics,
        output_dir=output_dir,
    )

    return {
        "best_state": best_state,
        "temperature": calibrated_temp,
        "test_metrics": test_metrics,
        "cal_ece": cal_ece,
        "total_params": total_params,
        "size_kb": size_kb,
        "artifacts": artifact_paths,
    }


if __name__ == "__main__":
    train_agripulse()
