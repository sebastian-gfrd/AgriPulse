"""Tests for Market Price Parity Engine and WFP food price integration."""

import pytest
from agripulse.market import MarketParityEngine, NegotiationAnalysis


def test_market_engine_benchmarks():
    engine = MarketParityEngine(data_path="data/wfp_market_prices.json")
    ref = engine.get_benchmark("coffee_parchment_grade1")

    assert ref.currency == "KES"
    assert ref.benchmark_price > 100.0
    assert ref.fair_floor_price < ref.benchmark_price
    assert ref.predatory_threshold < ref.fair_floor_price


def test_predatory_offer_detection():
    engine = MarketParityEngine(data_path="data/wfp_market_prices.json")

    # Predatory low offer (70 KES vs 125 KES benchmark)
    analysis_low = engine.evaluate_middleman_offer(70.0, "coffee_parchment_grade1", estimated_volume_kg=500.0)
    assert analysis_low.is_predatory is True
    assert analysis_low.price_gap_amount == 55.0
    assert analysis_low.total_potential_loss == 27500.0
    assert "USIKUBALI" in analysis_low.advice_swahili

    # Fair offer (122 KES vs 125 KES benchmark)
    analysis_fair = engine.evaluate_middleman_offer(122.0, "coffee_parchment_grade1", estimated_volume_kg=500.0)
    assert analysis_fair.is_predatory is False
    assert "NZURI" in analysis_fair.advice_swahili
