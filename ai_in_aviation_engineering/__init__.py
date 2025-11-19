"""Utilities for the Ai-in-aviation-engineering project."""

from .flight_capacity import (
    OverbookingAssessment,
    assess_overbooking,
    average_passengers,
)
from .io_utils import dedup_names, is_potential_multi_index

__all__ = [
    "dedup_names",
    "is_potential_multi_index",
    "average_passengers",
    "assess_overbooking",
    "OverbookingAssessment",
]
