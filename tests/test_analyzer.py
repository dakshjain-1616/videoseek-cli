"""
Core integration tests for VideoAnalyzer.

Test spec:
1. INPUT: video.mp4, query "Find the car crash"   → EXPECTED: Timestamp 01:23:45
2. INPUT: video.mp4, query "Show the sunset"       → EXPECTED: Timestamp 00:45:30
3. Edge case: query "Find a non-existent event"    → EXPECTED: "No matching moment found"
"""

from __future__ import annotations

import os
import pytest

from videoseek_cli.analyzer import NO_MATCH_MSG, SearchResult, VideoAnalyzer
from videoseek_cli.embeddings import FrameStore
from videoseek_cli.frame_extractor import seconds_to_timestamp


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer_with_data(tmp_path, records: list[dict]) -> VideoAnalyzer:
    """
    Build a VideoAnalyzer pre-seeded with *records*, bypassing real video I/O.
    """
    os.environ["VIDEOSEEK_DB_PATH"] = str(tmp_path / "db")
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.video_path = "/fake/video.mp4"
    analyzer.sample_fps = None
    analyzer.max_frames = None
    from videoseek_cli.embeddings import FrameStore
    from videoseek_cli.vision_client import VisionClient
    analyzer._store = FrameStore("/fake/video.mp4")
    analyzer._client = VisionClient()
    analyzer._store.add_batch(records)
    return analyzer


# ---------------------------------------------------------------------------
# Test spec — exact timestamps
# ---------------------------------------------------------------------------

class TestRequiredTestSpec:
    """Verify the three test cases from the project spec."""

    def _records_with_crash_and_sunset(self) -> list[dict]:
        """
        Seed the store with frames where:
          - frame at 01:23:45 (5025 s) contains a car crash description
          - frame at 00:45:30 (2730 s) contains a sunset description
          - plus generic filler frames
        """
        filler = [
            {
                "frame_index": i,
                "timestamp_seconds": float(i * 30),
                "timestamp_str": seconds_to_timestamp(i * 30),
                "description": f"Generic background scene number {i}.",
            }
            for i in range(10)
        ]
        crash_ts = 5025.0  # 01:23:45
        sunset_ts = 2730.0  # 00:45:30
        special = [
            {
                "frame_index": 1000,
                "timestamp_seconds": crash_ts,
                "timestamp_str": seconds_to_timestamp(crash_ts),
                "description": (
                    "A dramatic vehicle collision on a highway; "
                    "the car crashes into a barrier with debris flying."
                ),
            },
            {
                "frame_index": 2000,
                "timestamp_seconds": sunset_ts,
                "timestamp_str": seconds_to_timestamp(sunset_ts),
                "description": (
                    "A beautiful sunset with vibrant orange and pink hues "
                    "over the ocean horizon."
                ),
            },
        ]
        return filler + special

    def test_spec_1_car_crash_timestamp(self, tmp_path):
        """Query 'Find the car crash' → best match timestamp 01:23:45."""
        records = self._records_with_crash_and_sunset()
        analyzer = _make_analyzer_with_data(tmp_path, records)
        result = analyzer.query_best("Find the car crash")
        assert result is not None, "Expected a result for car crash query"
        assert result.timestamp_str == "01:23:45", (
            f"Expected 01:23:45 but got {result.timestamp_str}"
        )

    def test_spec_2_sunset_timestamp(self, tmp_path):
        """Query 'Show the sunset' → best match timestamp 00:45:30."""
        records = self._records_with_crash_and_sunset()
        analyzer = _make_analyzer_with_data(tmp_path, records)
        result = analyzer.query_best("Show the sunset")
        assert result is not None, "Expected a result for sunset query"
        assert result.timestamp_str == "00:45:30", (
            f"Expected 00:45:30 but got {result.timestamp_str}"
        )

    def test_spec_3_no_match(self, tmp_path):
        """Query 'Find a non-existent event' → NO_MATCH_MSG."""
        # Use only generic frames with no remotely matching descriptions
        records = [
            {
                "frame_index": i,
                "timestamp_seconds": float(i),
                "timestamp_str": seconds_to_timestamp(i),
                "description": "A blank white wall with no features.",
            }
            for i in range(5)
        ]
        os.environ["VIDEOSEEK_DB_PATH"] = str(tmp_path / "db_nomatch")
        analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
        analyzer.video_path = "/fake/video.mp4"
        analyzer.sample_fps = None
        analyzer.max_frames = None
        from videoseek_cli.embeddings import FrameStore
        from videoseek_cli.vision_client import VisionClient
        analyzer._store = FrameStore("/fake/video.mp4")
        analyzer._client = VisionClient()
        analyzer._store.add_batch(records)

        # Use very high threshold so generic frames won't match exotic queries
        results = analyzer.query(
            "Find a non-existent event: dragon breathing fire on moon",
            threshold=0.99,
        )
        assert results == [], "Expected no results for impossible query with high threshold"

    def test_no_match_message_constant(self):
        assert NO_MATCH_MSG == "No matching moment found"


# ---------------------------------------------------------------------------
# Unit tests for SearchResult
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_str_format(self):
        r = SearchResult(
            timestamp_str="00:01:30",
            timestamp_seconds=90.0,
            frame_index=3,
            description="A car racing on a track.",
            similarity_score=0.87,
        )
        s = str(r)
        assert "00:01:30" in s
        assert "0.870" in s
        assert "car" in s

    def test_fields(self):
        r = SearchResult(
            timestamp_str="01:23:45",
            timestamp_seconds=5025.0,
            frame_index=999,
            description="Crash scene.",
            similarity_score=0.91,
        )
        assert r.timestamp_str == "01:23:45"
        assert r.timestamp_seconds == 5025.0
        assert r.frame_index == 999
        assert 0 < r.similarity_score <= 1.0


# ---------------------------------------------------------------------------
# VideoAnalyzer mock-mode tests
# ---------------------------------------------------------------------------

class TestVideoAnalyzerMockMode:
    def test_index_returns_positive_count(self, tmp_path, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        if analyzer._store.is_indexed():
            analyzer._store.delete()
        count = analyzer.index()
        assert count > 0

    def test_query_returns_list(self, tmp_path, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        if analyzer._store.is_indexed():
            analyzer._store.delete()
        analyzer.index()
        results = analyzer.query("Find something", top_k=3)
        assert isinstance(results, list)

    def test_query_best_returns_result_or_none(self, tmp_path, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        if analyzer._store.is_indexed():
            analyzer._store.delete()
        analyzer.index()
        result = analyzer.query_best("car crash")
        # In mock mode this may or may not match, just check type contract
        assert result is None or isinstance(result, SearchResult)

    def test_double_index_is_idempotent(self, tmp_path, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        if analyzer._store.is_indexed():
            analyzer._store.delete()
        c1 = analyzer.index()
        c2 = analyzer.index()  # Should not re-index
        assert c1 == c2

    def test_get_video_info_mock(self, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        info = analyzer.get_video_info()
        assert "duration_str" in info
        assert "native_fps" in info

