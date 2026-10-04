# SYSTEM SPECIFICATION: AgriPulse
**Target Event:** World Bank Small AI for Development Hackathon (Seoul Summit 2026)  
**Track:** Annex B: Agriculture & Smallholder Resilience  
**Target Persona:** Noor (Ondera Highlands, Coffee Smallholder)  
**System Class:** Small AI / Edge-Compiled Time-Series Engine & Bimodal Advisory  
**Compute Runtime:** Pure CPU (JAX + Flax + XLA)  

---

## 1. Context & Problem Statement

### 1.1 Persona Constraints (Noor)
* **Demographics & Livelihood:** Noor is a 38-year-old smallholder farmer cultivating 2 hectares in the Ondera highlands (Arabica coffee on upper slope; maize and beans below). She has been a member of the Ondera Coffee Cooperative for 11 years.
* **Device Ecology:** 
  * *Primary device:* A basic feature phone (keypad) used for voice calls, SMS, and mobile money. During field hours, the phone remains at the house.
  * *Secondary device:* A smartphone owned by her 16-year-old daughter, who boards at school during weekdays. The smartphone is only accessible on weekends when the daughter returns home.
* **Connectivity & Energy:** Zero Wi-Fi at home. Internet access relies entirely on intermittent, prepaid 3G data bundles purchased on weekends.
* **Linguistic Context:** Primary communication in local language (e.g., Swahili); national language used only when strictly required. Low digital and screen literacy.
* **Institutional Bottlenecks:**
  1. *Extension Void:* The public extension agent visited her sub-county only twice in the entire past year.
  2. *Cryptic Yield Loss:* Coffee yields have dropped without clear diagnostics (invisible early-stage fungal incubation).
  3. *Predatory Market Asymmetry:* Middlemen arrive at harvest with pick-up trucks and dictate arbitrary purchase prices because Noor lacks an independent market reference.

---

## 2. Project Description: AgriPulse

**AgriPulse** is an ultra-lightweight, zero-bandwidth agricultural decision support system designed to anticipate biological crop threats and eliminate commercial asymmetry for smallholders in remote settings.

Rather than relying on image scanning of damaged leaves (which detects infection only after irreversible damage has occurred), AgriPulse utilizes a **1D Temporal Convolutional Neural Network (1D-CNN)** compiled with JAX/XLA to detect microclimatic incubation patterns of Coffee Leaf Rust (*Hemileia vastatrix*) **15 to 25 days before visual symptom eruption**. 

The system operates under a **Bimodal Dispatch Topology**, bridging offline high-performance edge compute with ultra-austere SMS/voice delivery.

```
       [WEEKEND: Smartphone + Intermittent 3G]
                         │
     Download 30-day compressed telemetry (~5 KB)
     (NASA POWER agro-climate + CHIRPS rainfall)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │     AgriPulse Edge Engine    │
        │  XLA-Compiled 1D-CNN on CPU      │
        │  Footprint: < 80 KB | Lat: < 2ms │
        └────────────────┬─────────────────┘
                         │
        Evaluates Risk & Calibrates Uncertainty
        Fetches WFP Regional Price Baseline
                         │
                         ▼
       [WEEKDAYS: Feature Phone Delivery / SMS]
                         │
        Zero-cost cultural actions + Fair price alert
        dispatched in local language (Swahili) to Noor
```

---

## 3. Architecture Design Motivations & Rationale

### 3.1 Why Temporal 1D-CNN over 2D Computer Vision (Leaf Scanning)
* **Biological Latency:** When a farmer photographs an orange rust pustule on a leaf, the fungal cycle has already completed sporulation. At that stage, control requires costly systemic fungicides that smallholders cannot afford. A 1D-CNN evaluates temporal environmental series ($T_{max}$, $T_{min}$, relative humidity, rain accumulation, solar radiation) to detect the exact 48–72 hour window when fungal spores penetrate stomata, enabling zero-cost cultural interventions.
* **Field Failure Modes of Vision:** Real-world field photos suffer from dirty camera lenses, background clutter, and variable sunlight, leading to shortcut learning (collapsing accuracy from 99% in studio datasets like PlantVillage to <35% in actual fields). Time-series climate metrics from satellites are physically grounded and invariant to optical noise.

### 3.2 Why Pure CPU Execution with JAX / Flax
* **Elimination of Cloud & GPU Overhead:** General deep learning frameworks (PyTorch, TensorFlow) carry runtime footprints of hundreds of megabytes. AgriPulse is built in Flax and compiled via XLA into static, vectorized C/machine code.
* **Hardware Inclusivity:** The entire model has fewer than 25,000 parameters (<80 KB binary). Inference executes in under 2 ms on low-end ARM Cortex-A53 cores (budget Android phones or Raspberry Pi nodes) without requiring specialized neural accelerators or constant network polling.

### 3.3 Why Bimodal Ecosystem Alignment
* **Reality of the Household:** A tool that requires a smartphone in the field fails Noor, as the smartphone leaves every Monday morning with her daughter. AgriPulse schedules heavy edge-compute and telemetry caching during the weekend window, caching weekly guidance and scheduling lightweight SMS alerts to Noor’s keypad phone during the week.

### 3.4 Integration of Market Parity (WFP Food Prices)
* **Countering Informational Reductionism:** Pure agronomic advice does not solve smallholder poverty if harvested yields are sold at a loss. Integrating the WFP Food Prices dataset provides an independent local price benchmark, giving Noor direct bargaining power when the middleman arrives.

### 3.5 Calibrated Uncertainty & Human-in-the-Loop Guardrail (Pass/Fail Compliance)
* **Ethical AI Mandate:** A false negative leads to crop destruction; a false positive causes unnecessary labor. The system implements **Temperature Scaling** on model logits:
  * **$P \ge 0.75$ (High Risk):** Triggers specific cultural defense actions.
  * **$0.40 < P < 0.75$ (Uncertainty Zone):** The system strictly abstains from automated advice and displays the mandatory safeguard: *"Hali ya hewa si ya kawaida. Wasiliana na afisa ugani au mkuu wa ushirika"* (Ambiguous pattern. Consult the cooperative extension officer).
  * **$P \le 0.40$ (Low Risk):** Normal farm maintenance.

---

## 4. Technical Architecture Specifications

### 4.1 Input Tensor Specification
The input tensor $\mathbf{X}$ represents a continuous sliding window of 30 days across 6 normalized physical channels:
$$\mathbf{X} \in \mathbb{R}^{\text{Batch} \times 30 \times 6}$$

| Channel | Variable Source | Physical Dimension | Biological Relevance |
| :--- | :--- | :--- | :--- |
| 0 | CHIRPS | Daily Precipitation ($mm$) | Washing of spores / surface moisture |
| 1 | NASA POWER | Daily $T_{max}$ ($^\circ\text{C}$) | Thermal threshold of germination |
| 2 | NASA POWER | Daily $T_{min}$ ($^\circ\text{C}$) | Nocturnal condensation trigger |
| 3 | Derived | $\Delta T = T_{max} - T_{min}$ | Microclimatic thermal amplitude |
| 4 | NASA POWER | Relative Humidity Mean ($\%$) | Fungal spore survival ($>85\%$ required) |
| 5 | iSDAsoil | Static Soil Metric (pH / Clay ratio) | Basal physiological host vulnerability |

### 4.2 Flax Linen Model Architecture (`src/model.py`)
```python
import os
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import flax.linen as nn

class AgriPulse1DCNN(nn.Module):
    num_classes: int = 1

    @nn.compact
    def __call__(self, x, train: bool = False):
        # x shape: (batch, 30, 6)
        
        # Stage 1: Short-term microclimatic pattern extraction (3-5 days)
        h = nn.Conv(features=16, kernel_size=(5,), padding='SAME')(x)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.gelu(h)
        h = nn.max_pool(h, window_shape=(2,), strides=(2,))  # (batch, 15, 16)

        # Stage 2: Residual block for medium-term cumulative dynamics
        res = nn.Conv(features=32, kernel_size=(1,), padding='SAME')(h)
        h_res = nn.Conv(features=32, kernel_size=(3,), padding='SAME')(h)
        h_res = nn.BatchNorm(use_running_average=not train)(h_res)
        h_res = nn.gelu(h_res)
        h_res = nn.Conv(features=32, kernel_size=(3,), padding='SAME')(h_res)
        h_res = nn.BatchNorm(use_running_average=not train)(h_res)
        h = nn.gelu(h_res + res)
        h = nn.max_pool(h, window_shape=(2,), strides=(2,))  # (batch, 7, 32)

        # Stage 3: Long-term biological incubation integration
        h = nn.Conv(features=64, kernel_size=(3,), padding='SAME')(h)
        h = nn.BatchNorm(use_running_average=not train)(h)
        h = nn.gelu(h)

        # Temporal collapse via Global Average Pooling
        h = jnp.mean(h, axis=1)  # (batch, 64)

        # Decision Head
        h = nn.Dense(features=32)(h)
        h = nn.gelu(h)
        logits = nn.Dense(features=self.num_classes)(h)
        return logits.squeeze(-1)
```

### 4.3 Calibrated Inference & Guardrails (`src/engine.py`)
```python
import jax
import jax.numpy as jnp

TEMPERATURE = 1.25  # Pre-calibrated via validation temperature scaling

@jax.jit
def predict_rust_risk(params, batch_stats, input_window):
    """
    input_window: preprocessed array of shape (1, 30, 6)
    Returns:
        risk_score: float in [0.0, 1.0]
        status_code: 0=STABLE, 1=UNCERTAIN (HUMAN_IN_THE_LOOP), 2=CRITICAL_ACTION
    """
    logits = model.apply(
        {'params': params, 'batch_stats': batch_stats},
        input_window,
        train=False
    )
    calibrated_prob = jax.nn.sigmoid(logits / TEMPERATURE).squeeze()
    
    # Branching guardrail logic
    status_code = jnp.where(
        calibrated_prob >= 0.75,
        2,  # CRITICAL
        jnp.where(calibrated_prob <= 0.40, 0, 1)  # 0=STABLE, 1=UNCERTAIN
    )
    return calibrated_prob, status_code
```

### 4.4 Frugal SMS Payload Specification (`src/telemetry.py`)
For cooperative-wide aggregation or 2G network transmission, telemetry is serialized into an **18-byte binary frame**:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Protocol (0xCA)|          Unix Timestamp (uint32)             |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
| Risk (uint8)  | Status (uint8)|       Farmer ID (uint16)      |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
|                   GeoHash (8 ASCII characters)                |
|                                                               |
+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
```

---

## 5. Implementation Roadmap for Gemini Antigravity-CLI

When developing the MVP repository during the competition weekend, follow this pipeline:

1. **Step 1: Ingestion & Mock Synthesis (`data/`):**
   * Generate synthetic 30-day multivariate time series mimicking historical CHIRPS and NASA POWER data for Ondera highland coordinates.
   * Format input matrix to shape `(N, 30, 6)` with binary ground truth labels based on epidemiological degree-day rust criteria.
2. **Step 2: Flax Model Pipeline (`src/`):**
   * Implement `AgriPulse1DCNN` in Flax Linen.
   * Write training loop using `optax.adam` and weighted binary cross-entropy with logits.
   * Export trained weights (`params` and `batch_stats`) via `flax.serialization`.
3. **Step 3: Edge Benchmarking (`tests/`):**
   * Ensure `os.environ["JAX_PLATFORMS"] = "cpu"` is active.
   * Benchmark warm-up vs compiled execution time (verify latency $< 5\text{ ms}$).
4. **Step 4: Bimodal Interface Simulator (`cli/`):**
   * Implement interactive CLI demo: simulates weekend 3G sync, triggers JAX CPU inference, and displays resulting Swahili text message formatted for a basic feature phone.