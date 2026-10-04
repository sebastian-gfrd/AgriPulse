# Contributing to AgriPulse

Thank you for your interest in contributing to **AgriPulse: Ultra-Frugal Edge AI for Smallholder Farmers**.

This project was built for the **World Bank Youth Summit 2026** and **Hack-Nation Challenge 04 (Small AI for Development)**. We welcome contributions from agronomists, machine learning researchers, agricultural extensionists, and edge systems engineers.

---

## Frugal AI Core Principles

All contributions to AgriPulse must adhere to three fundamental design constraints:

1. **Ultra-Compact Footprint (<80 KB):**
   Models must run on low-cost ARM Cortex-A or feature phone processors without discrete GPUs or neural accelerators. FP16 binaries must stay strictly below 80 KB.

2. **Ethical Guardrails & Deferral:**
   If prediction confidence falls within the uncertainty margin ($\tau_1 \le p \le \tau_2$), the system **must** defer to human cooperative agricultural extension officers rather than generating false advice.

3. **Bimodal Dispatch & Offline First:**
   Zero assumption of home broadband or continuous Wi-Fi. Heavy downloads are restricted to intermittent weekend synchronization; daily operation relies strictly on 18-byte binary GSM frames or 160-character SMS.

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/AgriPulse.git
   cd AgriPulse
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Run the automated test suite:**
   ```bash
   pytest -v tests/
   ```

4. **Run the edge benchmark:**
   ```bash
   python3 -m agripulse.benchmark
   ```

---

## Submitting Pull Requests

1. Fork the repo and create your feature branch: `git checkout -b feature/my-new-feature`
2. Ensure all 17 tests pass: `pytest -v tests/`
3. Commit your changes with conventional commit messages: `git commit -m 'feat: add agro-climatic transferability for cocoa'`
4. Push to your branch and open a Pull Request.

---

## Code of Conduct

We are committed to providing a friendly, safe, and welcoming environment for everyone, particularly smallholders and marginalized farming communities who inspire this work.
