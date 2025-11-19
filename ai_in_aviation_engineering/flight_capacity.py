"""Flight capacity helpers.

This module provides a very small helper that can be used to compute the
average number of passengers in historical flights and to determine whether a
flight can be safely overbooked based on that history.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def average_passengers(passenger_counts: Sequence[float]) -> float:
    """Return the average passenger count.

    Args:
        passenger_counts: Historical passenger counts for previous flights.

    Raises:
        ValueError: If ``passenger_counts`` is empty.
    """

    if not passenger_counts:
        raise ValueError("at least one passenger count is required")
    return sum(passenger_counts) / len(passenger_counts)


@dataclass(frozen=True)
class OverbookingAssessment:
    """Represents the overbooking status of a flight."""

    average: float
    seat_capacity: int
    safety_margin: float

    @property
    def can_overbook(self) -> bool:
        """Whether there is enough margin to allow overbooking."""

        threshold = self.seat_capacity * (1 - self.safety_margin)
        return self.average <= threshold


def assess_overbooking(
    passenger_counts: Sequence[float],
    seat_capacity: int,
    safety_margin: float = 0.05,
) -> OverbookingAssessment:
    """Assess whether a flight can be overbooked.

    Args:
        passenger_counts: Historical passenger counts for previous flights.
        seat_capacity: Number of seats available on the flight.
        safety_margin: Portion of seats to keep as a buffer. A value of ``0.05``
            means we keep 5% of the seats as a buffer and only overbook if the
            historical average is below 95% of the capacity.

    Returns:
        An :class:`OverbookingAssessment` instance.
    """

    if seat_capacity <= 0:
        raise ValueError("seat_capacity must be positive")
    if not 0 <= safety_margin < 1:
        raise ValueError("safety_margin must be between 0 and 1")

    avg = average_passengers(passenger_counts)
    return OverbookingAssessment(avg, seat_capacity, safety_margin)
