"""Telemetry serialization, SMS payload encoding, and voice script generation.

Implements the 18-byte binary frame specification for low-bandwidth 2G/USSD uplink
and provides GSM-compliant (<160 chars) SMS messages in Swahili for Noor's keypad phone.
"""

from dataclasses import dataclass
import struct
import time
from typing import Optional, Tuple


PROTOCOL_MAGIC = 0xCA  # AgriPulse telemetry header byte


def compute_crc8(data: bytes) -> int:
    """Standard 8-bit CRC with polynomial 0x07."""
    crc = 0x00
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


@dataclass
class TelemetryFrame:
    """18-byte frugal telemetry packet for 2G / cellular uplink."""

    timestamp: int
    risk_score_pct: int  # 0 to 100
    status_code: int  # 0: Stable, 1: Uncertain, 2: Critical
    farmer_id: int  # uint16 (0 to 65535)
    geohash: str  # 8 characters ASCII

    def pack(self) -> bytes:
        """Serialize into 18 bytes binary buffer."""
        # Truncate or pad geohash to exactly 8 characters
        geo_bytes = self.geohash[:8].ljust(8, "0").encode("ascii")
        # Header (1b) + Timestamp (4b) + Risk (1b) + Status (1b) + FarmerID (2b) + GeoHash (8b) = 17b
        payload = struct.pack(
            ">B I B B H 8s",
            PROTOCOL_MAGIC,
            self.timestamp,
            self.risk_score_pct,
            self.status_code,
            self.farmer_id,
            geo_bytes,
        )
        crc = compute_crc8(payload)
        return payload + struct.pack("B", crc)

    @classmethod
    def unpack(cls, buffer: bytes) -> "TelemetryFrame":
        """Deserialize from 18-byte buffer with CRC verification."""
        if len(buffer) != 18:
            raise ValueError(f"Expected 18 bytes frame, got {len(buffer)}")

        payload = buffer[:17]
        expected_crc = buffer[17]
        actual_crc = compute_crc8(payload)
        if expected_crc != actual_crc:
            raise ValueError(f"CRC check failed! Expected {expected_crc}, calculated {actual_crc}")

        magic, timestamp, risk, status, farmer_id, geo_bytes = struct.unpack(">B I B B H 8s", payload)
        if magic != PROTOCOL_MAGIC:
            raise ValueError(f"Invalid protocol magic byte: {hex(magic)} (expected {hex(PROTOCOL_MAGIC)})")

        return cls(
            timestamp=timestamp,
            risk_score_pct=risk,
            status_code=status,
            farmer_id=farmer_id,
            geohash=geo_bytes.decode("ascii"),
        )


def format_sms_message(
    status_code: int,
    risk_score: float,
    coffee_price: float = 125.0,
    language: str = "swahili",
) -> Tuple[str, int]:
    """Generate concise, actionable SMS strictly within the 160-character GSM limit.

    Returns:
        (message_text, character_count)
    """
    risk_pct = int(round(risk_score * 100))

    if language.lower() == "swahili":
        if status_code == 2:  # Critical
            msg = (
                f"[AgriPulse] HATARI KUTU: {risk_pct}%. Punguza matawi na kivuli cha miti "
                f"ndani ya siku 5 kuzuia ugonjwa. Bei kahawa: KES {coffee_price:.0f}/kg."
            )
        elif status_code == 1:  # Uncertain / Guardrail
            msg = (
                f"[AgriPulse] TAHADHARI: Hali ya hewa si ya kawaida ({risk_pct}%). "
                f"Wasiliana na afisa ugani au mkuu wa ushirika. Usipulize dawa bila ushauri."
            )
        else:  # Stable
            msg = (
                f"[AgriPulse] SHAMBA SALAMA: Hatari {risk_pct}%. Hakuna dalili za kutu. "
                f"Endelea na palizi ya kawaida. Bei kahawa: KES {coffee_price:.0f}/kg."
            )
    else:  # English
        if status_code == 2:
            msg = (
                f"[AgriPulse] RUST ALERT: {risk_pct}%. Prune canopy and shade trees within 5 days "
                f"to stop spore spread. Coffee price: KES {coffee_price:.0f}/kg."
            )
        elif status_code == 1:
            msg = (
                f"[AgriPulse] UNCERTAIN: Atypical weather ({risk_pct}%). "
                f"Consult coop extension officer before spraying or taking action."
            )
        else:
            msg = (
                f"[AgriPulse] CROP SAFE: Rust risk {risk_pct}%. "
                f"Continue regular field maintenance. Coffee price: KES {coffee_price:.0f}/kg."
            )

    return msg, len(msg)


def format_voice_script(
    status_code: int,
    risk_score: float,
    coffee_price: float = 125.0,
    farmer_name: str = "Noor",
    language: str = "swahili",
) -> str:
    """Generate spoken IVR / voice advisory script for low-literacy farmers."""
    risk_pct = int(round(risk_score * 100))

    if language.lower() == "swahili":
        if status_code == 2:
            return (
                f"Habari mama {farmer_name}. Huu ni mfumo wa AgriPulse kutoka Ushirika wa Ondera. "
                f"Katika siku thelathini zilizopita, unyevu na joto la milimani limefikia kiwango cha hatari "
                f"cha asilimia {risk_pct} cha kuzalisha ukungu wa kutu ya majani. "
                f"Ili kuzuia majani kuanguka bila kutumia gharama za dawa, nenda shambani wiki hii upunguze matawi "
                f"ya juu na kivuli cha miti ili upepo na jua vipenye. "
                f"Pia, bei ya soko ya kahawa ya WFP leo ni shilingi {coffee_price:.0f} kwa kilo. "
                f"Usikubali bei ya chini kutoka kwa madalali wa barabarani."
            )
        elif status_code == 1:
            return (
                f"Habari mama {farmer_name}. Mfumo wa AgriPulse umeona mabadiliko yasiyo ya kawaida "
                f"kwenye hali ya hewa ya milimani. Kiwango cha uhakika ni asilimia {risk_pct}. "
                f"Mfumo hauwezi kuthibitisha hatari moja kwa moja. Tafadhali wasiliana na afisa ugani Bwana Mwangi "
                f"au mkuu wa kituo cha ushirika cha Ondera akague shamba lako kabla ya kufanya uamuzi wowote."
            )
        else:
            return (
                f"Habari mama {farmer_name}. Huu ni mfumo wa AgriPulse. Shamba lako la kahawa kwenye mteremko wa juu "
                f"liko salama. Hatari ya ugonjwa wa kutu ni ndogo, asilimia {risk_pct}. "
                f"Endelea na palizi na usafi wa kawaida. Bei ya ushirika wa kahawa leo ni shilingi {coffee_price:.0f} kwa kilo."
            )
    else:
        if status_code == 2:
            return (
                f"Hello {farmer_name}. This is AgriPulse from Ondera Cooperative. "
                f"Over the last 30 days, humidity and temperature have reached a {risk_pct}% risk level "
                f"for coffee leaf rust incubation. To protect your crop at zero cost, thin the upper tree canopy "
                f"this week to allow sunlight and airflow to dry the leaves. "
                f"Today's official cooperative coffee price is KES {coffee_price:.0f} per kilogram. "
                f"Do not accept discounted prices from opportunistic middlemen."
            )
        elif status_code == 1:
            return (
                f"Hello {farmer_name}. AgriPulse detected ambiguous microclimatic conditions ({risk_pct}%). "
                f"Rather than guessing, we recommend speaking with your local cooperative extension officer "
                f"before taking field actions."
            )
        else:
            return (
                f"Hello {farmer_name}. Your coffee plot is stable. Rust risk is low ({risk_pct}%). "
                f"Continue routine weeding and farm sanitation. Benchmark coffee price is KES {coffee_price:.0f} per kg."
            )
