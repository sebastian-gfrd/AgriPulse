"""Cooperative Extensionist Aggregation & Epidemic Heatmap Engine for AgriPulse.

Simulates the Ondera Coffee Cooperative central terminal receiving 18-byte binary
telemetry frames from 100 smallholders across the Aberdares / Ondera highlands.
Performs spatial clustering, outbreak boundary detection, and targeted extension dispatch.
"""

from dataclasses import dataclass, asdict
import json
import os
import time
from typing import Dict, List, Tuple
import numpy as np

from agripulse.telemetry import TelemetryFrame, PROTOCOL_MAGIC


@dataclass
class SmallholderReport:
    """Decompressed cooperative member status record."""

    farmer_id: int
    farmer_name: str
    latitude: float
    longitude: float
    elevation_m: int
    crop_hectares: float
    risk_score_pct: int
    status_code: int
    status_label: str
    reported_timestamp: int
    geohash: str


class CooperativeTerminal:
    """Central cooperative intelligence node for public extensionist Bwana Mwangi."""

    def __init__(self, coop_name: str = "Ondera Farmers Cooperative Society"):
        self.coop_name = coop_name
        self.center_lat = -0.420
        self.center_lon = 36.950
        self.reports: List[SmallholderReport] = []

    def generate_simulated_member_uplinks(
        self,
        num_farmers: int = 80,
        seed: int = 42,
    ) -> List[bytes]:
        """Simulate 80 smallholders transmitting 18-byte telemetry frames over 2G network."""
        rng = np.random.default_rng(seed)
        frames = []

        # Include Noor explicitly as Farmer 1042
        noor_frame = TelemetryFrame(
            timestamp=int(time.time()),
            risk_score_pct=82,
            status_code=2,
            farmer_id=1042,
            geohash="kzdp4v8q",
        ).pack()
        frames.append(noor_frame)

        # Generate neighbor frames along the mountain ridge
        for i in range(1, num_farmers):
            # Spatial distribution along the slopes: altitude 1600m to 1950m
            lat_offset = rng.normal(0, 0.035)
            lon_offset = rng.normal(0, 0.030)

            # Risk correlated with elevation and micro-valley moisture
            is_humid_valley = lat_offset < -0.01
            base_risk = rng.uniform(65, 95) if is_humid_valley else rng.uniform(10, 55)
            risk_pct = int(np.clip(base_risk + rng.normal(0, 8), 0, 100))

            if risk_pct >= 75:
                status = 2
            elif risk_pct <= 40:
                status = 0
            else:
                status = 1  # Uncertain guardrail zone

            farmer_id = 1000 + i
            geohash = f"kzd{abs(int(lat_offset*1000)):02d}{abs(int(lon_offset*1000)):03d}"[:8]

            frame = TelemetryFrame(
                timestamp=int(time.time() - rng.uniform(60, 86400)),
                risk_score_pct=risk_pct,
                status_code=status,
                farmer_id=farmer_id,
                geohash=geohash,
            ).pack()
            frames.append(frame)

        return frames

    def ingest_telemetry_batch(self, raw_frames: List[bytes]) -> Dict[str, any]:
        """Ingest, verify CRC8, and index 18-byte binary frames."""
        self.reports.clear()
        valid_count = 0
        corrupted_count = 0

        rng = np.random.default_rng(42)

        for buf in raw_frames:
            try:
                frame = TelemetryFrame.unpack(buf)
                valid_count += 1

                # Decode spatial coordinates from geohash / offset
                # (For demo simulation, map farmer ID to realistic spatial coordinate on slope)
                if frame.farmer_id == 1042:
                    name = "Noor Chemutai (Upper Slope)"
                    lat = self.center_lat + 0.008
                    lon = self.center_lon - 0.005
                    elev = 1850
                    ha = 2.0
                else:
                    name = f"Member #{frame.farmer_id}"
                    lat = self.center_lat + rng.normal(0, 0.025)
                    lon = self.center_lon + rng.normal(0, 0.025)
                    elev = int(1750 + (lat - self.center_lat) * 2000)
                    ha = round(float(rng.uniform(0.8, 3.5)), 1)

                status_label = {0: "STABLE", 1: "UNCERTAIN_ESCALATION", 2: "CRITICAL_ACTION"}.get(
                    frame.status_code, "UNKNOWN"
                )

                self.reports.append(
                    SmallholderReport(
                        farmer_id=frame.farmer_id,
                        farmer_name=name,
                        latitude=round(lat, 5),
                        longitude=round(lon, 5),
                        elevation_m=elev,
                        crop_hectares=ha,
                        risk_score_pct=frame.risk_score_pct,
                        status_code=frame.status_code,
                        status_label=status_label,
                        reported_timestamp=frame.timestamp,
                        geohash=frame.geohash,
                    )
                )
            except ValueError:
                corrupted_count += 1

        # Summary statistics
        total = len(self.reports)
        crit_count = sum(1 for r in self.reports if r.status_code == 2)
        uncert_count = sum(1 for r in self.reports if r.status_code == 1)
        stable_count = sum(1 for r in self.reports if r.status_code == 0)
        total_ha_at_risk = sum(r.crop_hectares for r in self.reports if r.status_code == 2)

        summary = {
            "cooperative_name": self.coop_name,
            "total_uplinks_processed": total,
            "frames_corrupted": corrupted_count,
            "bandwidth_consumed_bytes": total * 18,
            "bandwidth_consumed_kb": round((total * 18) / 1024.0, 2),
            "outbreak_epidemiology": {
                "critical_action_farmers": crit_count,
                "uncertain_escalation_farmers": uncert_count,
                "stable_farmers": stable_count,
                "critical_pct": round((crit_count / max(total, 1)) * 100, 1),
                "total_hectares_threatened": round(total_ha_at_risk, 1),
            },
            "extensionist_dispatch_priority": [
                asdict(r) for r in sorted(self.reports, key=lambda x: x.risk_score_pct, reverse=True)[:10]
            ],
        }

        return summary

    def export_geojson(self, output_path: str = "data/cooperative_epidemic_map.geojson"):
        """Export spatial risk layer for GIS / Leaflet web visualization."""
        features = []
        for r in self.reports:
            color = "#dc2626" if r.status_code == 2 else ("#eab308" if r.status_code == 1 else "#16a34a")
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [r.longitude, r.latitude],
                },
                "properties": {
                    "farmer_id": r.farmer_id,
                    "farmer_name": r.farmer_name,
                    "elevation_m": r.elevation_m,
                    "hectares": r.crop_hectares,
                    "risk_pct": r.risk_score_pct,
                    "status_code": r.status_code,
                    "status_label": r.status_label,
                    "marker_color": color,
                },
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "metadata": {
                "cooperative": self.coop_name,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            "features": features,
        }

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)

        return output_path
