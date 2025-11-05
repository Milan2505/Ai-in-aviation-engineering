import pytest

from ai_in_aviation_engineering import io_utils
from ai_in_aviation_engineering.io_utils import dedup_names, is_potential_multi_index


def test_is_potential_multi_index_requires_tuples():
    columns = [("a", "b"), ("c", "d")]
    assert is_potential_multi_index(columns)


def test_is_potential_multi_index_with_multiindex_instance():
    columns = io_utils.MultiIndex([(1, 2), (3, 4)])
    assert not is_potential_multi_index(columns)


def test_is_potential_multi_index_respects_index_col_names():
    columns = ["idx", ("a", "b"), ("c", "d")]
    assert is_potential_multi_index(columns, index_col=["idx"])
    assert not is_potential_multi_index(columns, index_col=["a"])


def test_is_potential_multi_index_handles_unhashable_index_col():
    columns = [("a", "b"), ("c", "d")]
    index_col = [["ignore"]]
    assert is_potential_multi_index(columns, index_col=index_col)


def test_dedup_names_basic_usage():
    names = ["x", "y", "x", "x"]
    assert dedup_names(names, is_potential_multiindex=False) == [
        "x",
        "y",
        "x.1",
        "x.2",
    ]


def test_dedup_names_multiindex_labels():
    names = [("a", "b"), ("a", "b")]
    assert dedup_names(names, is_potential_multiindex=True) == [
        ("a", "b"),
        ("a", "b.1"),
    ]


def test_dedup_names_preserves_existing_suffixes():
    names = ["col", "col.1", "col"]
    assert dedup_names(names, is_potential_multiindex=False) == [
        "col",
        "col.1",
        "col.2",
    ]


def test_dedup_names_does_not_mutate_input():
    names = ("x", "x")
    dedup_names(names, is_potential_multiindex=False)
    assert names == ("x", "x")
