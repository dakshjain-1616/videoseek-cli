"""Tests for the vision client and mock description generation."""

from __future__ import annotations

import os

import pytest

from videoseek_cli.vision_client import (
    VisionClient,
    _mock_description,
    mock_descriptions_for_query,
)


class TestMockDescription:
    def test_returns_string(self):
        desc = _mock_description("00:00:05")
        assert isinstance(desc, str)
        assert len(desc) > 5

    def test_deterministic(self):
        d1 = _mock_description("00:01:00")
        d2 = _mock_description("00:01:00")
        assert d1 == d2

    def test_different_timestamps_can_differ(self):
        d1 = _mock_description("00:00:00")
        d2 = _mock_description("00:00:07")
        # They may or may not differ but both should be valid strings
        assert isinstance(d1, str) and isinstance(d2, str)


class TestMockDescriptionsForQuery:
    def test_returns_correct_count(self):
        records = mock_descriptions_for_query("car crash", 10)
        assert len(records) == 10

    def test_crash_query_injects_crash_description(self):
        records = mock_descriptions_for_query("car crash", 30)
        descs = [r["description"] for r in records]
        # At least one frame should mention a collision/crash
        found = any(
            any(kw in d.lower() for kw in ["crash", "collision", "vehicle"])
            for d in descs
        )
        assert found, "Expected at least one crash-related description for 'car crash' query"

    def test_sunset_query_injects_sunset(self):
        records = mock_descriptions_for_query("beautiful sunset", 30)
        descs = [r["description"] for r in records]
        found = any(
            any(kw in d.lower() for kw in ["sunset", "hues", "golden", "dusk"])
            for d in descs
        )
        assert found

    def test_records_have_required_keys(self):
        records = mock_descriptions_for_query("test", 5)
        for r in records:
            assert "frame_index" in r
            assert "timestamp_seconds" in r
            assert "timestamp_str" in r
            assert "description" in r

    def test_timestamp_str_format(self):
        records = mock_descriptions_for_query("test", 5)
        for r in records:
            parts = r["timestamp_str"].split(":")
            assert len(parts) == 3


class TestVisionClientMockMode:
    def test_describe_frame_returns_string_in_mock_mode(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        client = VisionClient()
        desc = client.describe_frame("fake_b64", "00:00:01")
        assert isinstance(desc, str)
        assert len(desc) > 0

    def test_batch_describe_returns_correct_count(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        client = VisionClient()
        frames = [
            {
                "index": i,
                "timestamp_seconds": float(i),
                "timestamp_str": f"00:00:{i:02d}",
                "image_b64": "fakeb64",
            }
            for i in range(3)
        ]
        results = client.batch_describe(frames)
        assert len(results) == 3
        for r in results:
            assert "description" in r
            assert "timestamp_str" in r

    def test_batch_describe_calls_progress_callback(self):
        os.environ.pop("OPENROUTER_API_KEY", None)
        client = VisionClient()
        frames = [
            {
                "index": i,
                "timestamp_seconds": float(i),
                "timestamp_str": f"00:00:{i:02d}",
                "image_b64": "fakeb64",
            }
            for i in range(4)
        ]
        calls = []
        client.batch_describe(frames, progress_callback=lambda d, t: calls.append((d, t)))
        assert len(calls) == 4
        assert calls[-1] == (4, 4)
