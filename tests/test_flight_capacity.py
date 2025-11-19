import pytest

from ai_in_aviation_engineering import (
    OverbookingAssessment,
    assess_overbooking,
    average_passengers,
)


def test_average_passengers_basic():
    assert average_passengers([100, 110, 90]) == pytest.approx(100)


def test_average_passengers_requires_values():
    with pytest.raises(ValueError):
        average_passengers([])


def test_assess_overbooking_flags_safe_margin():
    assessment = assess_overbooking([80, 90, 85], seat_capacity=120, safety_margin=0.1)
    assert isinstance(assessment, OverbookingAssessment)
    assert assessment.can_overbook is True


def test_assess_overbooking_detects_risk():
    assessment = assess_overbooking([115, 118, 120], seat_capacity=120, safety_margin=0.05)
    assert assessment.can_overbook is False
