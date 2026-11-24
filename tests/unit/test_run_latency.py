from __future__ import annotations

import pytest

from polima.cli.run import latency_summary


def test_latency_summary_reports_release_percentiles():
    summary = latency_summary([1.0, 2.0, 3.0, 4.0, 5.0])
    assert summary["min"] == 1.0
    assert summary["max"] == 5.0
    assert summary["mean"] == 3.0
    assert summary["p50"] == 3.0
    assert summary["p95"] == pytest.approx(4.8)
    assert summary["p99"] == pytest.approx(4.96)
