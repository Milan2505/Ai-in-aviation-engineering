"""Optimised utilities inspired by :mod:`pandas.io.common`.

This module keeps the same public API as the small subset of pandas helpers
that we rely on while rewriting the hot paths in pure Python so they are easy
to experiment with inside this kata repository.  The implementations aim to be
compatible with the behaviour of the upstream helpers while applying a couple
of micro-optimisations that remove avoidable allocations when dealing with
large header definitions.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Sequence
from typing import DefaultDict, Tuple

try:  # pragma: no cover - pandas is not an installation requirement here.
    from pandas.core.indexes.api import MultiIndex  # type: ignore
except Exception:  # pragma: no cover - fallback used in the exercises.
    class MultiIndex(tuple):
        """Fallback MultiIndex implementation used for type comparisons."""

        pass

__all__ = ["dedup_names", "is_potential_multi_index"]


def _normalise_index_col(
    index_col: bool | Sequence[Hashable] | None,
) -> tuple[Sequence[Hashable], Iterable[Hashable]]:
    """Return both the original order and an efficient lookup container."""
    if index_col is None or isinstance(index_col, bool):
        return (), ()

    ordered = tuple(index_col)
    if not ordered:
        return ordered, ordered

    try:
        return ordered, frozenset(ordered)
    except TypeError:
        # ``index_col`` may contain unhashable entries (e.g. lists) when the
        # caller passes in bespoke descriptors.  Falling back to the tuple keeps
        # the semantics identical to ``list(index_col)`` used in pandas.
        return ordered, ordered


def _is_index_label(
    lookup: Iterable[Hashable],
    ordered: Sequence[Hashable],
    value: Hashable,
) -> bool:
    """Robust membership test that mirrors pandas' original semantics."""
    try:
        return value in lookup
    except TypeError:
        # ``lookup`` can be a ``frozenset`` containing hashable values, in which
        # case ``value`` raising ``TypeError`` means it is an unhashable tuple.
        # Falling back to the ordered sequence keeps behaviour identical to the
        # reference implementation.
        return value in ordered


def is_potential_multi_index(
    columns: Sequence[Hashable] | MultiIndex,
    index_col: bool | Sequence[Hashable] | None = None,
) -> bool:
    """Check whether the provided ``columns`` can be turned into a MultiIndex.

    The upstream pandas implementation repeatedly casts ``index_col`` to a
    ``list`` inside a comprehension.  When working with large inputs that is an
    avoidable allocation.  We pre-compute both the original order and (when
    possible) a hashed lookup container and reuse them during the membership
    checks.
    """

    if isinstance(columns, MultiIndex) or len(columns) == 0:
        return False

    ordered_index, lookup = _normalise_index_col(index_col)
    return all(
        isinstance(value, tuple)
        for value in columns
        if not _is_index_label(lookup, ordered_index, value)
    )


def _make_deduped_label(
    label: Hashable,
    suffix: int,
    is_potential_multiindex: bool,
) -> Hashable:
    if is_potential_multiindex:
        if not isinstance(label, tuple):  # pragma: no cover - defensive branch.
            raise TypeError("MultiIndex labels are expected to be tuples.")
        head: Tuple[Hashable, ...] = label[:-1]
        tail = f"{label[-1]}.{suffix}"
        return head + (tail,)
    return f"{label}.{suffix}"


def dedup_names(
    names: Sequence[Hashable],
    is_potential_multiindex: bool,
) -> Sequence[Hashable]:
    """Rename duplicate columns by appending an incrementing suffix.

    The pandas version keeps bumping counters while mutating the lookup dict in
    place.  That approach re-runs the same dictionary lookups when several
    collisions happen in a row.  Here we remember the latest suffix used for the
    original key and only create new candidates when necessary.
    """

    result = list(names)
    counts: DefaultDict[Hashable, int] = defaultdict(int)

    for pos, original in enumerate(result):
        occurrence = counts[original]
        if occurrence == 0:
            counts[original] = 1
            continue

        suffix = occurrence
        candidate = _make_deduped_label(original, suffix, is_potential_multiindex)
        while counts[candidate]:
            suffix += 1
            candidate = _make_deduped_label(original, suffix, is_potential_multiindex)
        result[pos] = candidate
        counts[original] = suffix + 1
        counts[candidate] = 1

    return result
