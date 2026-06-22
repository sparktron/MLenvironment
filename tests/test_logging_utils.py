"""Tests for append_metrics_csv schema validation."""

from __future__ import annotations

import csv

import pytest

from rl_framework.utils.logging_utils import CsvSchemaError, append_metrics_csv


def test_first_write_creates_header(tmp_path):
    path = tmp_path / "metrics.csv"
    append_metrics_csv(path, {"a": 1, "b": 2})
    rows = list(csv.reader(path.open()))
    assert rows[0] == ["a", "b"]
    assert rows[1] == ["1", "2"]


def test_matching_schema_appends(tmp_path):
    path = tmp_path / "metrics.csv"
    append_metrics_csv(path, {"a": 1, "b": 2})
    append_metrics_csv(path, {"a": 3, "b": 4})
    rows = list(csv.reader(path.open()))
    assert len(rows) == 3  # header + 2 data rows
    assert rows[2] == ["3", "4"]


def test_added_key_raises(tmp_path):
    path = tmp_path / "metrics.csv"
    append_metrics_csv(path, {"a": 1})
    with pytest.raises(CsvSchemaError, match="new keys not in header"):
        append_metrics_csv(path, {"a": 1, "b": 2})


def test_removed_key_raises(tmp_path):
    path = tmp_path / "metrics.csv"
    append_metrics_csv(path, {"a": 1, "b": 2})
    with pytest.raises(CsvSchemaError, match="header keys missing from data"):
        append_metrics_csv(path, {"a": 1})


def test_reordered_keys_raise(tmp_path):
    """Column order is part of the schema — DictWriter would misalign values."""
    path = tmp_path / "metrics.csv"
    append_metrics_csv(path, {"a": 1, "b": 2})
    with pytest.raises(CsvSchemaError):
        append_metrics_csv(path, {"b": 2, "a": 1})


def test_precreated_empty_file_gets_header(tmp_path):
    path = tmp_path / "metrics.csv"
    path.touch()
    append_metrics_csv(path, {"a": 1, "b": 2})
    rows = list(csv.reader(path.open()))
    assert rows[0] == ["a", "b"]
