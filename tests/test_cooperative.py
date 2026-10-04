"""Tests for Cooperative Terminal and Outbreak Aggregation."""

import os
import pytest
from agripulse.cooperative import CooperativeTerminal


def test_cooperative_terminal_aggregation(tmp_path):
    term = CooperativeTerminal()
    frames = term.generate_simulated_member_uplinks(num_farmers=30)
    assert len(frames) == 30

    summary = term.ingest_telemetry_batch(frames)
    assert summary["total_uplinks_processed"] == 30
    assert summary["bandwidth_consumed_bytes"] == 30 * 18
    assert summary["outbreak_epidemiology"]["critical_action_farmers"] >= 0

    out_geojson = str(tmp_path / "test_map.geojson")
    exported = term.export_geojson(out_geojson)
    assert os.path.exists(exported)
