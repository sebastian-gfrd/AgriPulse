"""AgriPulse Full-Stack Interactive Edge Server.

Runs a zero-dependency HTTP server providing REST APIs for edge inference
and serving an interactive mobile/desktop application with:
1. Nokia 105 retro feature phone simulator with real Swahili audio voice playback
2. Daughter's smartphone edge analytics with 30-day agroclimate Chart.js graphs
3. Extensionist cooperative epidemic map with Leaflet.js
4. WFP Food Prices market parity calculator
"""

import http.server
import json
import os
import socketserver
import urllib.parse
from typing import Any, Dict

import numpy as np

from agripulse.engine import AgriPulseEdgeEngine
from agripulse.market import MarketParityEngine
from agripulse.telemetry import format_sms_message, format_voice_script
from agripulse.cooperative import CooperativeTerminal


PORT = int(os.environ.get("PORT", 8080))
WEB_DIR = os.path.join(os.path.dirname(__file__), "web")

# Pre-load edge engines
print("Initializing AgriPulse Edge Engine...")
edge_engine = AgriPulseEdgeEngine()
market_engine = MarketParityEngine()
coop_terminal = CooperativeTerminal()

# Generate cooperative member uplinks
raw_frames = coop_terminal.generate_simulated_member_uplinks(num_farmers=80)
coop_summary = coop_terminal.ingest_telemetry_batch(raw_frames)
coop_terminal.export_geojson(os.path.join("data", "cooperative_epidemic_map.geojson"))

# Load test dataset
data = np.load("data/ondera_coffee_rust_dataset.npz")
X_test_raw = data["X_test_raw"]
y_test = data["y_test"]


class AgriPulseHandler(http.server.SimpleHTTPRequestHandler):
    """Custom HTTP handler serving API endpoints and web UI."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.path.abspath("."), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if parsed.path == "/":
            self.path = "/web/index.html"
            return super().do_GET()

        # API: Edge Prediction
        if parsed.path == "/api/predict":
            self._handle_predict(params)
            return

        # API: Cooperative Aggregation
        if parsed.path == "/api/cooperative":
            self._handle_cooperative()
            return

        # API: Market Parity Calculation
        if parsed.path == "/api/market":
            self._handle_market(params)
            return

        # API: Hardware Benchmarks
        if parsed.path == "/api/benchmark":
            self._handle_benchmark()
            return

        return super().do_GET()

    def _handle_predict(self, params):
        status_target = params.get("status", ["critical"])[0]

        # Select corresponding test sample
        sample_idx = 0
        if status_target == "critical":
            # Search for status_code == 2
            for i in range(len(X_test_raw)):
                if edge_engine.predict(X_test_raw[i]).status_code == 2:
                    sample_idx = i
                    break
        elif status_target == "uncertain":
            for i in range(len(X_test_raw)):
                if edge_engine.predict(X_test_raw[i]).status_code == 1:
                    sample_idx = i
                    break
        else:  # stable
            for i in range(len(X_test_raw)):
                if edge_engine.predict(X_test_raw[i]).status_code == 0:
                    sample_idx = i
                    break

        sample = X_test_raw[sample_idx]
        res = edge_engine.predict(sample)

        sms_swahili, sms_sw_len = format_sms_message(res.status_code, res.risk_score, 125.0, "swahili")
        sms_english, sms_en_len = format_sms_message(res.status_code, res.risk_score, 125.0, "english")

        audio_name_map = {0: "stable", 1: "uncertain", 2: "critical"}
        status_name = audio_name_map.get(res.status_code, "stable")

        response = {
            "sample_index": sample_idx,
            "risk_score": res.risk_score,
            "risk_score_pct": int(round(res.risk_score * 100)),
            "status_code": res.status_code,
            "status_label": res.status_label,
            "latency_ms": res.inference_latency_ms,
            "requires_human_escalation": res.requires_human_escalation,
            "sms": {
                "swahili": sms_swahili,
                "swahili_length": sms_sw_len,
                "english": sms_english,
                "english_length": sms_en_len,
            },
            "audio_url": f"/audio/agripulse_{status_name}_swahili.mp3",
            "audio_url_english": f"/audio/agripulse_{status_name}_english.mp3",
            "climatic_drivers": res.climatic_drivers,
            "timeseries": {
                "days": list(range(1, 31)),
                "rainfall": [round(float(v), 2) for v in sample[:, 0]],
                "tmax": [round(float(v), 1) for v in sample[:, 1]],
                "tmin": [round(float(v), 1) for v in sample[:, 2]],
                "rh": [round(float(v), 1) for v in sample[:, 4]],
            },
        }

        self._send_json(response)

    def _handle_cooperative(self):
        geojson_path = "data/cooperative_epidemic_map.geojson"
        geojson_data = {}
        if os.path.exists(geojson_path):
            with open(geojson_path, "r", encoding="utf-8") as f:
                geojson_data = json.load(f)

        response = {
            "summary": coop_summary,
            "geojson": geojson_data,
        }
        self._send_json(response)

    def _handle_market(self, params):
        try:
            offer = float(params.get("offer", [72.0])[0])
            volume = float(params.get("volume", [650.0])[0])
        except ValueError:
            offer = 72.0
            volume = 650.0

        analysis = market_engine.evaluate_middleman_offer(
            offered_price=offer,
            commodity_key="coffee_parchment_grade1",
            estimated_volume_kg=volume,
        )

        response = {
            "commodity": analysis.commodity,
            "offered_price": analysis.offered_price,
            "benchmark_price": analysis.benchmark_price,
            "fair_floor_price": analysis.fair_floor_price,
            "currency": analysis.currency,
            "gap_amount": analysis.price_gap_amount,
            "gap_percent": analysis.price_gap_percent,
            "is_predatory": analysis.is_predatory,
            "total_loss": analysis.total_potential_loss,
            "advice_swahili": analysis.advice_swahili,
            "advice_english": analysis.advice_english,
        }
        self._send_json(response)

    def _handle_benchmark(self):
        bench_path = "models/benchmark_report.json"
        if os.path.exists(bench_path):
            with open(bench_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._send_json(data)
        else:
            self._send_json({"error": "Benchmark report not generated yet."})

    def _send_json(self, data: Dict[str, Any]):
        content = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)


def run_server(port: int = PORT):
    os.makedirs("web", exist_ok=True)
    server_address = ("", port)
    # Enable socket address reuse to prevent port binding conflicts
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(server_address, AgriPulseHandler) as httpd:
        print("=" * 70)
        print(f" AGRIPULSE INTERACTIVE WEB APP RUNNING ON PORT {port} ")
        print(f" Access URL: http://localhost:{port}")
        print("=" * 70)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server.")


if __name__ == "__main__":
    run_server()
