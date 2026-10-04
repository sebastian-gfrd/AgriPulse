"""Tests for Multi-Crop Replicability and Transfer Matrix."""

import os
import pytest
from agripulse.transfer import GLOBAL_TRANSFER_REGISTRY, export_transferability_dossier


def test_transfer_registry_and_dossier(tmp_path):
    assert "kenya_coffee_rust" in GLOBAL_TRANSFER_REGISTRY
    assert "cote_divoire_cocoa_blackpod" in GLOBAL_TRANSFER_REGISTRY
    assert "uganda_maize_fall_armyworm" in GLOBAL_TRANSFER_REGISTRY

    out_file = str(tmp_path / "transfer_test.json")
    dossier = export_transferability_dossier(out_file)

    assert os.path.exists(out_file)
    assert len(dossier["supported_tracks"]) == 3
