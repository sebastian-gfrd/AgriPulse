"""Tests for Telemetry Serialization, CRC8, and SMS Constraints."""

import pytest
from agripulse.telemetry import (
    TelemetryFrame,
    compute_crc8,
    format_sms_message,
    format_voice_script,
    PROTOCOL_MAGIC,
)


def test_telemetry_frame_pack_unpack():
    frame = TelemetryFrame(
        timestamp=1727980800,
        risk_score_pct=78,
        status_code=2,
        farmer_id=1042,
        geohash="kzdp4v8q",
    )
    buffer = frame.pack()
    assert len(buffer) == 18, f"Payload must be strictly 18 bytes, got {len(buffer)}"

    restored = TelemetryFrame.unpack(buffer)
    assert restored.timestamp == frame.timestamp
    assert restored.risk_score_pct == 78
    assert restored.status_code == 2
    assert restored.farmer_id == 1042
    assert restored.geohash == "kzdp4v8q"


def test_telemetry_frame_corruption_detection():
    frame = TelemetryFrame(
        timestamp=1727980800,
        risk_score_pct=45,
        status_code=1,
        farmer_id=2024,
        geohash="kzdp4v8q",
    )
    buffer = bytearray(frame.pack())

    # Corrupt one byte
    buffer[5] ^= 0xFF
    with pytest.raises(ValueError, match="CRC check failed"):
        TelemetryFrame.unpack(bytes(buffer))

    # Invalid length
    with pytest.raises(ValueError, match="Expected 18 bytes"):
        TelemetryFrame.unpack(bytes(buffer[:10]))


def test_sms_gsm_length_limits():
    """Verify that all generated SMS messages strictly satisfy GSM 160-char single SMS billing limit."""
    for status in [0, 1, 2]:
        for lang in ["swahili", "english"]:
            for risk in [0.15, 0.55, 0.85]:
                msg, length = format_sms_message(status, risk, coffee_price=125.0, language=lang)
                assert length <= 160, f"SMS too long ({length} chars): {msg}"
                assert len(msg.strip()) > 20
                if lang == "swahili":
                    assert "[AgriPulse]" in msg


def test_voice_script_generation():
    for status in [0, 1, 2]:
        script = format_voice_script(status, 0.80, 125.0, farmer_name="Noor", language="swahili")
        assert "Noor" in script
        assert "AgriPulse" in script
        assert len(script) > 50
