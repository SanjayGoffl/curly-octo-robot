"""Tests for treatment recommendations."""

import pytest
from src.inference.recommendations import (
    get_recommendations,
    get_all_recommendations,
    format_recommendations,
    APPLE_DISEASE_RECOMMENDATIONS
)


def test_get_recommendations_healthy():
    """Test getting recommendations for healthy apple."""
    rec = get_recommendations("Apple Healthy")
    assert rec is not None
    assert "treatment" in rec
    assert "prevention" in rec


def test_get_recommendations_scab():
    """Test getting recommendations for apple scab."""
    rec = get_recommendations("Apple Scab")
    assert rec is not None
    assert "fungicide" in rec["treatment"].lower()


def test_get_recommendations_black_rot():
    """Test getting recommendations for apple black rot."""
    rec = get_recommendations("Apple Black Rot")
    assert rec is not None
    assert "fungicide" in rec["treatment"].lower()


def test_get_recommendations_cedar_rust():
    """Test getting recommendations for apple cedar rust."""
    rec = get_recommendations("Apple Cedar Rust")
    assert rec is not None
    assert "cedar" in rec["prevention"].lower() or "juniper" in rec["prevention"].lower()


def test_get_recommendations_unknown():
    """Test getting recommendations for unknown disease."""
    rec = get_recommendations("Unknown Disease")
    assert rec is None


def test_get_all_recommendations():
    """Test getting all recommendations."""
    all_rec = get_all_recommendations()
    assert len(all_rec) == 4
    assert "Apple Healthy" in all_rec
    assert "Apple Scab" in all_rec
    assert "Apple Black Rot" in all_rec
    assert "Apple Cedar Rust" in all_rec


def test_format_recommendations_valid():
    """Test formatting recommendations for valid disease."""
    formatted = format_recommendations("Apple Scab")
    assert "Apple Scab" in formatted
    assert "Treatment:" in formatted
    assert "Prevention:" in formatted


def test_format_recommendations_invalid():
    """Test formatting recommendations for invalid disease."""
    formatted = format_recommendations("Unknown Disease")
    assert "agronomist" in formatted.lower()


def test_all_diseases_have_recommendations():
    """Test that all diseases have recommendations."""
    for disease in APPLE_DISEASE_RECOMMENDATIONS.keys():
        rec = get_recommendations(disease)
        assert rec is not None
        assert len(rec["treatment"]) > 0
        assert len(rec["prevention"]) > 0
