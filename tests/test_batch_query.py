"""Tests for VideoAnalyzer.batch_query and related new features."""

from __future__ import annotations

import os

import pytest

from videoseek_cli.analyzer import NO_MATCH_MSG, SearchResult, VideoAnalyzer
from videoseek_cli.embeddings import FrameStore
from videoseek_cli.frame_extractor import seconds_to_timestamp
from videoseek_cli.vision_client import VisionClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_analyzer(tmp_path, records: list[dict]) -> VideoAnalyzer:
    os.environ["VIDEOSEEK_DB_PATH"] = str(tmp_path / "db")
    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.video_path = "/fake/batch_video.mp4"
    analyzer.sample_fps = None
    analyzer.max_frames = None
    from videoseek_cli.analyzer import AnalysisStats
    analyzer.stats = AnalysisStats()
    analyzer._store = FrameStore("/fake/batch_video.mp4")
    analyzer._client = VisionClient()
    analyzer._store.add_batch(records)
    return analyzer


def _rich_records() -> list[dict]:
    filler = [
        {
            "frame_index": i,
            "timestamp_seconds": float(i * 10),
            "timestamp_str": seconds_to_timestamp(i * 10),
            "description": f"Generic background scene number {i}.",
        }
        for i in range(8)
    ]
    special = [
        {
            "frame_index": 100,
            "timestamp_seconds": 5025.0,
            "timestamp_str": "01:23:45",
            "description": "A dramatic car crash on a highway with debris flying.",
        },
        {
            "frame_index": 200,
            "timestamp_seconds": 2730.0,
            "timestamp_str": "00:45:30",
            "description": "A beautiful sunset with orange and pink hues over the ocean.",
        },
        {
            "frame_index": 300,
            "timestamp_seconds": 300.0,
            "timestamp_str": "00:05:00",
            "description": "Bright flames and fire engulfing a wooden structure.",
        },
    ]
    return filler + special


# ---------------------------------------------------------------------------
# batch_query tests
# ---------------------------------------------------------------------------

class TestBatchQuery:
    def test_returns_dict_with_all_queries(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        queries = ["Find the car crash", "Show the sunset", "Find fire"]
        result = analyzer.batch_query(queries, top_k=3)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(queries)

    def test_each_value_is_list(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        result = analyzer.batch_query(["Find the car crash"], top_k=3)
        assert isinstance(result["Find the car crash"], list)

    def test_car_crash_matched(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        result = analyzer.batch_query(["Find the car crash"], top_k=1, threshold=0.1)
        hits = result["Find the car crash"]
        assert hits, "Expected at least one result for car crash"
        assert hits[0].timestamp_str == "01:23:45"

    def test_sunset_matched(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        result = analyzer.batch_query(["Show the sunset"], top_k=1, threshold=0.1)
        hits = result["Show the sunset"]
        assert hits, "Expected at least one result for sunset"
        assert hits[0].timestamp_str == "00:45:30"

    def test_high_threshold_returns_empty(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        result = analyzer.batch_query(
            ["Find a unicorn dancing on the moon"], top_k=3, threshold=0.99
        )
        assert result["Find a unicorn dancing on the moon"] == []

    def test_multiple_queries_independently_ranked(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        result = analyzer.batch_query(
            ["Find the car crash", "Show the sunset"], top_k=1, threshold=0.1
        )
        crash_ts = result["Find the car crash"][0].timestamp_str if result["Find the car crash"] else None
        sunset_ts = result["Show the sunset"][0].timestamp_str if result["Show the sunset"] else None
        # The two queries should resolve to different best-match timestamps
        assert crash_ts != sunset_ts

    def test_empty_queries_list(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        result = analyzer.batch_query([], top_k=3)
        assert result == {}

    def test_single_query_same_as_query(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        single = analyzer.query("Find the car crash", top_k=3, threshold=0.1)
        batch = analyzer.batch_query(["Find the car crash"], top_k=3, threshold=0.1)
        batch_hits = batch["Find the car crash"]
        assert len(single) == len(batch_hits)
        for s, b in zip(single, batch_hits):
            assert s.timestamp_str == b.timestamp_str
            assert abs(s.similarity_score - b.similarity_score) < 1e-6


# ---------------------------------------------------------------------------
# export_frames tests
# ---------------------------------------------------------------------------

class TestExportFrames:
    def test_export_creates_files_in_mock_mode(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        results = analyzer.query("Find the car crash", top_k=1, threshold=0.1)
        assert results, "Need at least one result to export"
        out_dir = str(tmp_path / "exports")
        exported = analyzer.export_frames(results, output_dir=out_dir)
        assert len(exported) == len(results)
        for p in exported:
            from pathlib import Path
            assert Path(p).exists()

    def test_export_stub_contains_description(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        results = analyzer.query("Find the car crash", top_k=1, threshold=0.1)
        out_dir = str(tmp_path / "exports2")
        exported = analyzer.export_frames(results, output_dir=out_dir)
        from pathlib import Path
        content = Path(exported[0]).read_text()
        assert "Timestamp" in content or "Mock" in content

    def test_export_empty_results_returns_empty_list(self, tmp_path):
        analyzer = _make_analyzer(tmp_path, _rich_records())
        exported = analyzer.export_frames([], output_dir=str(tmp_path / "exports3"))
        assert exported == []


# ---------------------------------------------------------------------------
# AnalysisStats / get_stats tests
# ---------------------------------------------------------------------------

class TestAnalysisStats:
    def test_stats_keys_present(self, tmp_path, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        if analyzer._store.is_indexed():
            analyzer._store.delete()
            from videoseek_cli.analyzer import AnalysisStats
            analyzer.stats = AnalysisStats()
            analyzer._store = FrameStore("/fake/video.mp4")
        analyzer.index()
        stats = analyzer.get_stats()
        assert "frames_indexed" in stats
        assert "index_time_seconds" in stats
        assert "query_time_seconds" in stats
        assert "total_time_seconds" in stats

    def test_frames_indexed_positive(self, tmp_path, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        if analyzer._store.is_indexed():
            analyzer._store.delete()
            from videoseek_cli.analyzer import AnalysisStats
            analyzer.stats = AnalysisStats()
            analyzer._store = FrameStore("/fake/video.mp4")
        analyzer.index()
        stats = analyzer.get_stats()
        assert stats["frames_indexed"] > 0

    def test_total_time_is_sum(self, tmp_path, fresh_db_path):
        analyzer = VideoAnalyzer("/fake/video.mp4")
        if analyzer._store.is_indexed():
            analyzer._store.delete()
            from videoseek_cli.analyzer import AnalysisStats
            analyzer.stats = AnalysisStats()
            analyzer._store = FrameStore("/fake/video.mp4")
        analyzer.index()
        analyzer.query("something", top_k=1)
        stats = analyzer.get_stats()
        expected = round(
            stats["index_time_seconds"] + stats["query_time_seconds"], 3
        )
        assert abs(stats["total_time_seconds"] - expected) < 0.01
