# AgriPulse: Submission Dossier & Technical Whitepaper
**Challenge:** Hack-Nation Challenge 04 — Small AI for Development Hackathon  
**Organizers:** World Bank Group Digital & AI Vice Presidency &bull; Youth Summit &bull; DEC &bull; ITS  
**Showcase:** Global AI & Digital Summit (Seoul, October 2026)  
**Track:** Annex B: Agriculture & Smallholder Resilience  
**Target Persona:** Noor Chemutai (38, Ondera Highlands, Kenya)  
**Live Interactive Demo:** [https://agripulse-452096033898.us-central1.run.app](https://agripulse-452096033898.us-central1.run.app)  

---

## 1. Official Problem Statement Formula

> **"Because of AgriPulse, Noor and highland coffee smallholders will implement zero-cost canopy pruning and negotiate fair parchment prices by day 15 of fungal microclimatic incubation and harvest market arrival that they would otherwise suffer up to 40% cryptic yield loss and accept predatory 42% markdowns from middlemen; we know because public extension officers visit only twice a year and rural price indices in UN WFP/HDX reveal severe farm-gate price suppression."**

---

## 2. Executive Summary & Development Relevance (Score Weight: 20%)

More than **2.6 billion people** remain offline worldwide. In rural Sub-Saharan Africa, internet penetration drops to **27%**, and while 84% of women own basic mobile phones, only **31% own smartphones** (GSMA Mobile Gender Gap Report). Agricultural extension systems face severe personnel deficits; in Noor’s district, public extension officers visit only **twice a year**.

When fungal epidemics like Coffee Leaf Rust (*Hemileia vastatrix*) strike, smallholders experience **cryptic yield losses of up to 40%**. Conventional tech interventions fail because they prescribe **optical leaf-scanning apps**:
* **The Biological Fallacy of Computer Vision:** When an orange rust pustule is visible to a smartphone camera, the fungus has already completed stomatal penetration and sporulation. At this stage, crop rescue requires expensive systemic fungicides ($40–$60/ha) that smallholders cannot afford, driving rural households into debt.
* **The Device Ecology Fallacy:** Noor’s daughter boards at school during weekdays with the household's only smartphone. A tool requiring a smartphone in the field simply does not exist for Noor from Monday to Friday.

**AgriPulse** resolves this dilemma through a **Bimodal Dispatch Topology**:
1. **Weekend Sync:** Noor’s daughter downloads a compressed **4.8 KB telemetry package** (NASA POWER satellite observations and UN WFP market prices) via a prepaid 3G bundle.
2. **Edge Execution:** An ultra-lightweight **1D Temporal Convolutional Neural Network (1D-CNN)** executes on the phone's CPU in **0.18 ms**, evaluating 30-day multivariate microclimatic dynamics.
3. **Biological Anticipation:** Detects the invisible fungal germination window **15 to 25 days before visual symptoms erupt**.
4. **Weekday Feature Phone Delivery:** Dispatches actionable, zero-cost cultural interventions (canopy pruning, shade regulation, drainage) and independent WFP coffee price baselines via **standard GSM SMS (<160 chars)** and **synthesized Swahili voice audio** to Noor’s basic keypad phone.

---

## 3. Why Small AI? Algorithmic Value Proposition (Score Weight: 15%)

| Evaluation Question | Conventional Approach | AgriPulse (Small AI Approach) |
| :--- | :--- | :--- |
| **Why not a simple SMS rule ("If it rains, prune")?** | Weather is non-linear. Rain alone does not cause rust; germination requires **sustained continuous relative humidity $\ge 82\%$ for $>6$ hours** combined with an **optimal thermal window of $19.5^\circ\text{C}–25.5^\circ\text{C}$** and narrow thermal amplitude ($\Delta T$). Static rules trigger overwhelming false alarms during normal rains, inducing alert fatigue. | Evaluates a multi-channel temporal tensor $\mathbf{X} \in \mathbb{R}^{1 \times 30 \times 6}$ across 30 days, capturing non-linear biological degree-day accumulation with **ROC-AUC of 0.9931**. |
| **Why not 2D Computer Vision (PlantVillage)?** | High optical failure rate: real field photos suffer from dirty lenses, glare, and background foliage, dropping accuracy from 99% in studio conditions to $<35\%$ in the field. Sporulation is already irreversible. | Evaluates physically grounded atmospheric series from satellites, completely invariant to optical noise or camera quality. |
| **Why not a cloud LLM (ChatGPT / Claude)?** | Requires continuous 4G/5G broadband, high data bundle costs, massive cloud data center GPU clusters, and carries high hallucination risks. | Compiles via JAX/XLA into a static C-vectorized binary of **only 28.55 KB (FP16)**. Runs locally on pure CPU with zero server dependency. |

---

## 4. Empirical Data Grounding & Provenance (Score Weight: 15%)

In accordance with Section 7.2 of the World Bank Hackathon concept note, AgriPulse is built upon **100% real empirical data**:

1. **NASA POWER Agroclimatology (2015–2023):**
   * **Volume:** 23,009 real daily satellite observations covering 7 key Kenyan coffee-growing highland regions:
     * Ondera / Nyeri Highlands (-0.42° S, 36.95° E, ~1,850m) — Noor's farm
     * Kirinyaga (Kerugoya coffee belt)
     * Kiambu (Upper coffee highlands)
     * Murang'a (Gatanga coffee zone)
     * Embu (Runyenjes slopes)
     * Meru (Imenti South)
     * Kisii (Southwestern coffee highlands)
   * **Variables:** `PRECTOTCORR` (Precipitation mm), `T2M_MAX` (°C), `T2M_MIN` (°C), $\Delta T = T_{max} - T_{min}$, and `RH2M` (Relative Humidity %).
2. **UN WFP Food Prices via HDX (`wfp_food_prices_ken.csv`):**
   * **Volume:** 28,336 historical market records across 226 Kenyan markets providing empirical price distributions for cash and food crops (Arabica Coffee, White Maize, Dry Rosecoco Beans).
3. **iSDAsoil (30m Africa):**
   * High-resolution spatial mapping of acidic volcanic clay soil properties (pH 5.25–5.60, clay fraction 38–45%).

### Scored Criterion §7.2: What the Data Does NOT Cover & Gap Analysis
* **Spatial Smoothing:** Satellite grid cells (0.5° for NASA POWER) smooth out microclimatic variations between deep valley floors and exposed ridge tops.
* **Aspect Insolation:** Slope orientation (sunward vs. leeward) affects dew drying rates not captured in regional rasters.
* **Mitigation Strategy:** AgriPulse implements **Temperature Scaling Calibration ($T = 1.3906$)**. Whenever environmental series fall into the ambiguous boundary zone ($0.40 < P < 0.75$), the system strictly refrains from guessing and activates the **Human-in-the-Loop Guardrail**.

---

## 5. Responsible AI, Safety & Pass/Fail Guardrail (Pass/Fail Mandate)

> **World Bank Pass/Fail Requirement:** *"A confident wrong answer is costly when it touches someone's livelihood... each entry should incorporate a fail-safe: AI signposting to a decision-maker when the data it is acting on is not enough to give a definitive answer eg. 'not sure — ask a person' rather than guessing."*

```
                    [Calibrated Risk Probability P]
                                   │
         ┌─────────────────────────┼─────────────────────────┐
         ▼                         ▼                         ▼
     P <= 0.40               0.40 < P < 0.75              P >= 0.75
    [ STABLE ]            [ HUMAN-IN-THE-LOOP ]         [ CRITICAL ]
   Normal Farm           Ambiguous Weather Pattern     High Infection Risk
   Sanitation           Mandatory Signposting to       Zero-Cost Canopy
   & Weeding            Extension Officer Mwangi       Thinning within 5 Days
```

### Exact Guardrail Messages Formatted for Single GSM SMS:
* **Status 1 (Uncertainty Fail-Safe in Swahili — 131 chars):**  
  `[AgriPulse] TAHADHARI: Hali ya hewa si ya kawaida (64%). Wasiliana na afisa ugani au mkuu wa ushirika. Usipulize dawa bila ushauri.`
* **Status 1 (English Translation):**  
  `[AgriPulse] UNCERTAIN: Atypical weather (64%). Consult coop extension officer before spraying or taking action.`
* **Status 2 (Critical Outbreak in Swahili — 120 chars):**  
  `[AgriPulse] HATARI KUTU: 82%. Punguza matawi na kivuli cha miti ndani ya siku 5 kuzuia ugonjwa. Bei kahawa: KES 125/kg.`

---

## 6. Edge Hardware Benchmark & Small AI Fidelity (Score Weight: 25%)

Audited on **1,000 warm CPU inference cycles** using the real-data trained model:

| Metric | Hackathon Requirement | AgriPulse Result | Compliance Status |
| :--- | :---: | :---: | :---: |
| **Model Footprint (FP16 Binary)** | $< 80.0\text{ KB}$ | **28.55 KB** | **PASSED (64% safety margin)** |
| **Model Footprint (FP32 Binary)** | $< 80.0\text{ KB}$ | **57.08 KB** | **PASSED (29% safety margin)** |
| **Flax MsgPack Checkpoint** | $< 80.0\text{ KB}$ | **57.86 KB** | **PASSED (28% safety margin)** |
| **CPU Latency (Median p50)** | $< 2.0\text{ ms}$ | **0.165 ms** | **PASSED (12x faster)** |
| **CPU Latency (95th percentile)** | $< 2.0\text{ ms}$ | **0.249 ms** | **PASSED (8x faster)** |
| **CPU Throughput** | High efficiency | **5,472 inferences/sec** | **PASSED** |
| **Runtime GPU Dependency** | Zero at inference | **Zero (Pure CPU XLA)** | **PASSED** |
| **2G Telemetry Uplink Frame** | Frugal payload | **18 bytes (CRC8 protected)** | **PASSED** |
| **Single GSM SMS Length** | $\le 160\text{ chars}$ | **115 to 131 chars** | **PASSED** |
| **Zero-Dependency Micro-Runtime** | Embedded portability | **Pure NumPy engine verified (<1e-4 parity)** | **PASSED** |

---

## 7. Pan-African Scalability & Multi-Crop Generalization (Score Weight: 10%)

The core 1D-CNN temporal architecture is mathematically invariant to geography and crop. To scale to another agricultural challenge across the Global South, only the geographic coordinate and SMS language template require updating:

```
                                  [AgriPulse Universal Engine]
                                                 │
            ┌────────────────────────────────────┼────────────────────────────────────┐
            ▼                                    ▼                                    ▼
       KENYA (East Africa)             CÔTE D'IVOIRE (West Africa)            UGANDA (Great Lakes)
       Arabica Coffee                       Cacao (Cocoa)                         White Maize
   Hemileia vastatrix (Rust)         Phytophthora megakarya (Black Pod)    Spodoptera frugiperda (Armyworm)
   30-day incubation window              25-day incubation window              20-day degree-day window
   Swahili SMS & Voice Advisory          French/Baoulé SMS Advisory            Luganda/English SMS Advisory
```

### Verified Scaling Economics:
* **Model Retraining Time on GPU:** $\sim 15$ seconds on NVIDIA RTX 5070.
* **Storage Footprint on Phone:** $< 30\text{ KB}$ per crop profile.
* **Annual Compute Cost per Farmer:** $< \$0.002\text{ USD}$ (executed entirely on edge devices).

---

## 8. Multi-Modal Accessibility & Voice Inclusion (Score Weight: 15%)

For elderly smallholders or farmers with limited literacy, AgriPulse provides an integrated voice synthesis engine ([`agripulse/audio.py`](file:///home/sebastian/WorldBank/agripulse/audio.py)) that generates natural spoken advisories in Swahili:

> *"Habari mama Noor. Huu ni mfumo wa AgriPulse kutoka Ushirika wa Ondera. Katika siku thelathini zilizopita, unyevu na joto la milimani limefikia kiwango cha hatari cha asilimia 82 cha kuzalisha ukungu wa kutu ya majani. Ili kuzuia majani kuanguka bila kutumia gharama za dawa, nenda shambani wiki hii upunguze matawi ya juu na kivuli cha miti ili upepo na jua vipenye. Pia, bei ya soko ya kahawa ya WFP leo ni shilingi 125 kwa kilo. Usikubali bei ya chini kutoka kwa madalali wa barabarani."*

---

## 9. Deliverables & Demonstration Links

* **Interactive Full-Stack Web Application:** Run `python3 app.py` &rarr; open `http://localhost:8080`.
* **Interactive CLI Simulator:** `python3 -m cli.simulator --scenario all`.
* **Automated Hardware Benchmark Audit:** `python3 -m agripulse.benchmark`.
* **Unit & Integration Test Suite:** `pytest -v tests/` (17/17 tests passing).
* **Live Cloud Run Production URL:** [https://agripulse-452096033898.us-central1.run.app](https://agripulse-452096033898.us-central1.run.app).
