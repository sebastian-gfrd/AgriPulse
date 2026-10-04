# AgriPulse: Ultra-Frugal Edge Agro-Climatic Decision Engine

[![Live Demo](https://img.shields.io/badge/Google%20Cloud%20Run-Live%20Demo-4285F4?logo=googlecloud&logoColor=white)](https://agripulse-452096033898.us-central1.run.app)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB.svg?logo=python&logoColor=white)]()
[![Tests](https://img.shields.io/badge/Tests-17%2F17%20Passing-brightgreen.svg)]()
[![World Bank Youth Summit 2026](https://img.shields.io/badge/World%20Bank%20Summit-Seoul%202026-0072CE.svg)](https://www.worldbank.org)
[![Hack-Nation Challenge 04](https://img.shields.io/badge/Hack--Nation-Challenge%2004-orange.svg)](https://hack-nation.org)
[![Track](https://img.shields.io/badge/Track-Annex%20B%3A%20Agriculture-brightgreen.svg)]()
[![Pure CPU Inference](https://img.shields.io/badge/Edge%20Runtime-Pure%20CPU%20%3C0.25ms-blueviolet.svg)]()
[![Model Footprint](https://img.shields.io/badge/Model%20Size-28.5%20KB%20FP16%20%7C%2057%20KB%20FP32-success.svg)]()

> **"What does localizing AI development mean for you?"**  
> Localizing AI does not mean training multi-billion parameter models in hyper-scale cloud data centers. For a smallholder farmer in the Global South, **localized AI means an intelligence engine small enough to download over intermittent 3G in five seconds, light enough to run on an emulated feature phone CPU without a GPU in 0.2 milliseconds, grounded in hyper-local agro-climatology, and humble enough to say *"I am unsure, ask your cooperative extension officer"* before guessing with someone's livelihood.**

---

## 1. Problem Statement (World Bank Hackathon Formula)

> **"Because of AgriPulse, Noor and highland coffee smallholders will implement zero-cost canopy pruning and negotiate fair parchment prices by day 15 of fungal microclimatic incubation and harvest market arrival that they would otherwise suffer up to 40% cryptic yield loss and accept predatory 42% markdowns from middlemen; we know because public extension officers visit only twice a year and rural price indices in WFP/HDX reveal severe farm-gate price suppression."**

---

## 2. Target Persona & Operational Realities: Meet Noor

* **Location:** Ondera Highlands (Aberdares / Mt. Kenya coffee belt, elevation ~1,850m).
* **Livelihood:** 38-year-old smallholder farmer cultivating 2 hectares (Arabica coffee on upper slope; maize and beans below). Member of the Ondera Coffee Cooperative for 11 years.
* **Device Ecology:**
  * **Primary:** Basic keypad feature phone (used for calls, SMS, M-Pesa). Stays at home while she works the upper slopes.
  * **Secondary:** 16-year-old daughter's budget Android smartphone, accessible **only on weekends** when the daughter returns from boarding school.
* **Connectivity & Energy:** Zero Wi-Fi at home. Data access is strictly limited to prepaid 3G data bundles bought on weekends.
* **Linguistic Context:** Speaks local language (Swahili) at home. Low screen literacy.
* **Institutional Bottlenecks:**
  1. *Extension Void:* The public extension agent visited her sub-county only twice in the entire past year.
  2. *Cryptic Yield Loss:* Coffee yields dropped silently due to early-stage fungal rust incubation (*Hemileia vastatrix*).
  3. *Predatory Market Asymmetry:* Middlemen arrive at harvest with pickup trucks and dictate arbitrary purchase prices because Noor lacks an independent reference.

---

## 3. Why Small AI? Value Proposition vs. Simpler Tools & Vision

| Alternative Tool | Why It Fails Noor |
| :--- | :--- |
| **Leaf-Scanning Computer Vision (2D-CNN)** | **Biologically Too Late:** When orange pustules appear on the leaf, the fungal cycle has completed sporulation. At this point, control requires expensive synthetic fungicides Noor cannot afford. Furthermore, smartphone optical lenses suffer from field blur, variable lighting, and dirty cameras (collapsing accuracy from 99% in studio datasets like PlantVillage to <35% in real fields). |
| **Static SMS / Broadcast Alerts** | Fails to capture non-linear microclimatic interactions. Weather is dynamic; a rule like *"if it rains, prune"* generates overwhelming false alarms during regular wet spells, inducing alert fatigue. |
| **Spreadsheets / Google Search** | Assumes high digital literacy, continuous broadband connectivity, and active user prompting—none of which exist on Noor's mountain slope. |
| **AgriPulse (1D Temporal CNN)** | **Anticipates infection 15–25 days before visual symptoms:** Computes degree-days, thermal oscillation ($\Delta T$), and continuous relative humidity periods over 30 days to pinpoint the exact 48–72h spore germination window, enabling **zero-cost cultural actions (canopy thinning, shade regulation)**. |

---

## 4. System Architecture: The Bimodal Dispatch Topology

```
 [WEEKEND: Daughter's Smartphone + Intermittent 3G]
                         │
     Download 30-day compressed telemetry (~4.8 KB)
     (NASA POWER agro-climate + CHIRPS rainfall)
                         │
                         ▼
        ┌──────────────────────────────────┐
        │     AgriPulse Edge Engine    │
        │  XLA-Compiled 1D-CNN on CPU      │
        │  Footprint: < 58 KB | Lat: < 0.3ms│
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

### Dual-Profile Execution Engine
1. **GPU-Accelerated Training (Host Station):**
   * Accelerated by **JAX + CUDA on NVIDIA GeForce RTX 5070 (12 GB VRAM)**.
   * Trains on 15,000 multi-channel temporal sequences in under 16 seconds.
   * Performs validation temperature scaling calibration ($T = 0.8027$).
2. **Pure CPU Edge Inference (Edge Target):**
   * Uses `JAX_PLATFORMS=cpu` and XLA kernel compilation.
   * Eliminates GPU/CUDA runtime dependencies on edge devices.
   * **Sub-millisecond inference latency (p50: 0.18 ms, p95: 0.27 ms)**.

---

## 5. Model Serialization & Hardware Compression

To meet the strict **< 80 KB** binary file constraint for edge deployment over weak connections, AgriPulse implements a multi-format serialization engine:

| Artifact Format | File Path | File Size | Compliance (< 80 KB) |
| :--- | :--- | :---: | :---: |
| **Edge FP16 Binary** | `models/agripulse_edge_fp16.bin` | **28.55 KB** | **PASSED (64% margin)** |
| **Edge FP32 Binary** | `models/agripulse_edge_fp32.bin` | **57.08 KB** | **PASSED (29% margin)** |
| **Flax MsgPack State** | `models/agripulse_model.msgpack` | **57.86 KB** | **PASSED (28% margin)** |
| **Model Metadata JSON** | `models/agripulse_metadata.json` | 5.20 KB | Self-contained specs |

---

## 6. Datasets Grounding & Critical Gap Analysis

AgriPulse is built upon **100% real empirical data** ingested directly from the open official sources cited in the World Bank Concept Note:

1. **NASA POWER Agroclimatology (2015–2023):** 23,009 real daily satellite observations covering 7 major Kenyan coffee-growing highland regions:
   * Ondera / Nyeri Highlands (-0.42° S, 36.95° E, ~1,850m) — Noor's farm
   * Kirinyaga (Kerugoya coffee belt)
   * Kiambu (Upper coffee highlands)
   * Murang'a (Gatanga coffee zone)
   * Embu (Runyenjes slopes)
   * Meru (Imenti South)
   * Kisii (Southwestern coffee highlands)
   * Channels: `PRECTOTCORR` (Precipitation mm), `T2M_MAX`, `T2M_MIN`, $\Delta T$, `RH2M` (Relative Humidity %).
2. **UN WFP Food Prices via HDX (`wfp_food_prices_ken.csv`):** 28,336 historical market records across 226 Kenyan markets providing real-world price baselines for cash and staple crops (Coffee, White Maize, Dry Beans).
3. **iSDAsoil (30m Africa):** Gridded soil property index (acidic volcanic clay pH 5.25–5.60, clay fraction 38–45%).
4. **Masakhane / Common Voice:** Swahili linguistic grounding for SMS and voice synthesis.

### What the Data Does NOT Cover (Scored Hackathon Criterion §7.2)
* **Spatial Smoothing:** Satellite grid cells (0.5° for NASA POWER) smooth out hyper-local valley fogs and ridge-top microclimates.
* **Topographical Invariance:** Micro-slope aspects (sunward vs. leeward slopes) receive distinct solar radiation not reflected in coarse gridded reanalysis.
* **Mitigation Strategy:** AgriPulse compensates by incorporating **Temperature-Scaled Uncertainty Calibration ($T=1.3906$)**: whenever environmental inputs fall into the borderline ambiguity zone ($0.40 < P < 0.75$), the system strictly refrains from guessing and triggers human signposting.

---

## 7. Responsible AI & Guardrail Fail-Safe (Pass/Fail Mandate)

A confident wrong prediction is dangerous: a false negative ruins a harvest; a false positive induces unnecessary labor or chemical expense.

```
       [Calibrated Output Probability P]
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
   P <= 0.40     0.40 < P < 0.75    P >= 0.75
   [ STABLE ]     [ UNCERTAIN ]    [ CRITICAL ]
   Normal Farm    HUMAN-IN-THE-LOOP  Zero-cost cultural
   Sanitation     Mandatory Officer  canopy thinning &
                  Signposting        drainage action
```

### The Fail-Safe Signposting in Swahili:
> *"TAHADHARI: Hali ya hewa si ya kawaida (64%). Wasiliana na afisa ugani au mkuu wa ushirika kabla ya hatua yoyote."*  
> *(Unusual microclimatic pattern. Consult cooperative extension officer before taking any action.)*

---

## 8. Verified Edge Hardware Benchmark Results

Audited via `agripulse.benchmark` on 1,000 warm CPU inference cycles using the real-data trained model:

```
======================================================================
 AGRIPULSE: EDGE CPU BENCHMARK & HARDWARE COMPLIANCE AUDIT 
======================================================================
[+] MODEL BINARY SIZES:
  • Edge FP16 Binary:        28.55 KB  (Low-Bandwidth Mode)
  • Edge FP32 Binary:        57.08 KB  (Target: < 80.00 KB) -> PASSED
  • Flax MsgPack Checkpoint: 57.86 KB

[+] CPU INFERENCE LATENCY (1,000 RUNS ON SINGLE CORE):
  • Mean Latency:            0.183 ms
  • Median (p50):            0.165 ms
  • 95th Percentile:         0.249 ms  (Target: < 2.000 ms) -> PASSED
  • 99th Percentile:         0.296 ms
  • Peak Throughput:         5,472 inferences/second

[+] EMPIRICAL TEST PERFORMANCE (1,711 SATELLITE WINDOWS):
  • Accuracy:                95.68%
  • Recall (Sensitivity):    95.77%
  • ROC-AUC:                 0.9931
  • Calibrated ECE:          0.0084 (sub-1% calibration error)
======================================================================
```

---

## 9. Quickstart Guide

### Installation
```bash
git clone https://github.com/worldbank-youth-summit/agripulse.git
cd agripulse
pip install -e .
```

### 1. Run the Interactive User Journey Simulator
Simulate Noor's weekend sync, weekday SMS delivery, and market negotiation:
```bash
python3 -m cli.simulator --scenario all
```

### 2. Retrain the Model with GPU Acceleration
Leverage NVIDIA CUDA on RTX 5070:
```bash
python3 -m agripulse.train
```

### 3. Run Edge CPU Benchmarks
```bash
python3 -m agripulse.benchmark
```

### 4. Execute the Full Test Suite
```bash
pytest -v tests/
```

---

## 10. Repository Structure

```
/home/sebastian/WorldBank/
├── agripulse/
│   ├── __init__.py           # Package exports & version
│   ├── model.py              # Flax Linen 1D Temporal CNN architecture
│   ├── dataset.py            # Agroclimatic multivariate synthesizer & preprocessor
│   ├── train.py              # GPU training loop + validation temperature calibration
│   ├── serialize.py          # Weight compression to FP16/FP32 binary (<80 KB)
│   ├── engine.py             # Pure CPU edge inference engine & guardrail evaluator
│   ├── market.py             # WFP Food Prices parity & anti-predatory negotiation
│   ├── telemetry.py          # 18-byte binary frame codec & GSM SMS (<160 chars)
│   └── benchmark.py          # Automated latency, memory, and footprint benchmark
├── cli/
│   ├── __init__.py
│   └── simulator.py          # Interactive 4-scene terminal demonstration
├── data/
│   ├── dataset_metadata.json # Data provenance, licenses, and gap analysis
│   ├── norm_stats.json       # Normalization channel statistics
│   ├── ondera_coffee_rust_dataset.npz
│   └── wfp_market_prices.json# WFP / HDX commodity price references
├── models/
│   ├── agripulse_edge_fp16.bin   # 28.55 KB edge binary
│   ├── agripulse_edge_fp32.bin   # 57.08 KB edge binary
│   ├── agripulse_model.msgpack   # 57.86 KB Flax checkpoint
│   ├── agripulse_metadata.json   # Calibration T, thresholds, layer shapes
│   └── benchmark_report.json     # Hardware audit metrics
├── tests/
│   ├── test_model.py
│   ├── test_dataset.py
│   ├── test_serialization.py
│   ├── test_engine.py
│   ├── test_market.py
│   └── test_telemetry.py
├── pyproject.toml
├── requirements.txt
└── README.md
```
