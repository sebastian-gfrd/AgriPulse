"""Pure CPU Edge Inference Engine for AgriPulse.

Ensures zero dependency on cloud data centers, GPUs, or persistent internet.
Executes XLA-compiled 1D-CNN on CPU with sub-2ms latency and enforces
calibrated uncertainty guardrails with human-in-the-loop signposting.
"""

from dataclasses import dataclass
import json
import os
import time
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Force Pure CPU execution for edge runtime
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp

from agripulse.model import AgriPulse1DCNN
from agripulse.serialize import load_model_from_artifacts


@dataclass
class EdgePredictionResult:
    """Diagnostic outcome of an AgriPulse edge inference run."""

    risk_score: float  # Calibrated probability in [0.0, 1.0]
    status_code: int  # 0: STABLE, 1: UNCERTAIN (HUMAN_IN_THE_LOOP), 2: CRITICAL_ACTION
    status_label: str
    action_swahili: str
    action_english: str
    inference_latency_ms: float
    requires_human_escalation: bool
    climatic_drivers: Dict[str, float]


class AgriPulseEdgeEngine:
    """Self-contained edge inference engine running strictly on CPU."""

    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        self.model_dir = model_dir
        self.data_dir = data_dir

        # Load serialized parameters, batch_stats, temperature, and metadata
        self.params, self.batch_stats, self.temperature, self.metadata = (
            load_model_from_artifacts(model_dir=model_dir)
        )

        # Load normalization stats
        norm_path = os.path.join(data_dir, "norm_stats.json")
        if os.path.exists(norm_path):
            with open(norm_path, "r", encoding="utf-8") as f:
                norm_dict = json.load(f)
            self.means = np.array(norm_dict["means"], dtype=np.float32).reshape(1, 1, 6)
            self.stds = np.array(norm_dict["stds"], dtype=np.float32).reshape(1, 1, 6)
        else:
            self.means = np.zeros((1, 1, 6), dtype=np.float32)
            self.stds = np.ones((1, 1, 6), dtype=np.float32)

        self.model = AgriPulse1DCNN()
        
        # Define and JIT-compile static forward execution function on CPU
        def _forward_fn(x_norm: jnp.ndarray):
            logits = self.model.apply(
                {"params": self.params, "batch_stats": self.batch_stats},
                x_norm,
                train=False,
            )
            calibrated_prob = jax.nn.sigmoid(logits / self.temperature).squeeze()
            return logits, calibrated_prob

        self._jit_predict = jax.jit(_forward_fn)
        self._warmup_jit()

    def _warmup_jit(self):
        """Compile JIT graph on CPU during initialization."""
        dummy_x = jnp.zeros((1, 30, 6), dtype=jnp.float32)
        logits, prob = self._jit_predict(dummy_x)
        _ = prob.block_until_ready()

    def preprocess(self, raw_window: np.ndarray) -> np.ndarray:
        """Normalize raw input window of shape (30, 6) or (1, 30, 6)."""
        if raw_window.ndim == 2:
            raw_window = np.expand_dims(raw_window, axis=0)
        return (raw_window - self.means) / self.stds

    def extract_climatic_drivers(self, raw_window: np.ndarray) -> Dict[str, float]:
        """Summarize key physical drivers from the 30-day window."""
        if raw_window.ndim == 3:
            w = raw_window[0]
        else:
            w = raw_window

        total_precip = float(np.sum(w[:, 0]))
        max_tmax = float(np.max(w[:, 1]))
        min_tmin = float(np.min(w[:, 2]))
        mean_rh = float(np.mean(w[:, 4]))
        wet_days = int(np.sum(w[:, 0] > 1.5))

        return {
            "total_rainfall_mm": round(total_precip, 1),
            "wet_days_count": wet_days,
            "max_temperature_c": round(max_tmax, 1),
            "min_temperature_c": round(min_tmin, 1),
            "mean_relative_humidity": round(mean_rh, 1),
        }

    def predict(self, raw_window: np.ndarray) -> EdgePredictionResult:
        """Run calibrated inference and evaluate guardrails.

        Args:
            raw_window: Array of shape (30, 6) with real physical units:
                Channel 0: Rain (mm)
                Channel 1: Tmax (°C)
                Channel 2: Tmin (°C)
                Channel 3: Delta T (°C)
                Channel 4: RH (%)
                Channel 5: Soil metric

        Returns:
            EdgePredictionResult containing calibrated risk and actionable advice.
        """
        drivers = self.extract_climatic_drivers(raw_window)
        x_norm = self.preprocess(raw_window)

        start_time = time.perf_counter()
        _, prob_jax = self._jit_predict(jnp.array(x_norm))
        _ = prob_jax.block_until_ready()
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        prob_val = float(np.array(prob_jax))

        # Guardrail Decision Logic (Pass/Fail Human-in-the-Loop compliance)
        # Thresholds: P <= 0.40 -> Stable, 0.40 < P < 0.75 -> Uncertain, P >= 0.75 -> Critical
        if prob_val >= 0.75:
            status_code = 2
            status_label = "CRITICAL_ACTION"
            requires_human = False
            action_swahili = (
                "HATARI KUBWA YA KUTU YA MAJANI: Punguza matawi na kivuli cha miti mara moja "
                "kuruhusu mzunguko wa hewa. Hakikisha mitaro ya maji iko wazi."
            )
            action_english = (
                "HIGH RUST RISK: Thin canopy foliage and prune shade trees immediately to improve "
                "air circulation. Clear drainage ditches before spore eruption."
            )
        elif prob_val <= 0.40:
            status_code = 0
            status_label = "STABLE"
            requires_human = False
            action_swahili = (
                "SHAMBA LIKO SALAMA: Hakuna viashiria vya maambukizi ya kutu. "
                "Endelea na palizi na usafi wa kawaida wa shamba."
            )
            action_english = (
                "FARM STABLE: No microclimatic indicators of rust infection. "
                "Continue standard weed control and regular field sanitation."
            )
        else:
            status_code = 1
            status_label = "UNCERTAIN_HUMAN_IN_THE_LOOP"
            requires_human = True
            action_swahili = (
                "TAHADHARI YA HALI ISIYO YA KAWAIDA: Hali ya hewa si ya kawaida. "
                "Wasiliana na afisa ugani au mkuu wa ushirika kabla ya hatua yoyote."
            )
            action_english = (
                "AMBIGUOUS PATTERN: Environmental conditions show atypical variation. "
                "Consult the cooperative extension officer before taking action."
            )

        return EdgePredictionResult(
            risk_score=round(prob_val, 4),
            status_code=status_code,
            status_label=status_label,
            action_swahili=action_swahili,
            action_english=action_english,
            inference_latency_ms=round(latency_ms, 3),
            requires_human_escalation=requires_human,
            climatic_drivers=drivers,
        )
