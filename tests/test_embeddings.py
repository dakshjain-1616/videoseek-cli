"""Tests for ChromaDB-backed frame store."""

from __future__ import annotations

import os
import pytest

from videoseek_cli.embeddings import FrameStore


@pytest.fixture
def store(tmp_path):
    os.environ["VIDEOSEEK_DB_PATH"] = str(tmp_path / "db")
    return FrameStore("/fake/video.mp4")


class TestFrameStore:
    def test_empty_on_creation(self, store):
        assert not store.is_indexed()
        assert store.frame_count() == 0

    def test_add_single_frame(self, store):
        store.add_frame_description(
            frame_index=0,
            timestamp_seconds=0.0,
            timestamp_str="00:00:00",
            description="A car driving on a highway.",
        )
        assert store.is_indexed()
        assert store.frame_count() == 1

    def test_add_batch(self, store):
        records = [
            {
                "frame_index": i,
                "timestamp_seconds": float(i),
                "timestamp_str": f"00:00:{i:02d}",
                "description": f"Frame {i}: some visual content.",
            }
            for i in range(5)
        ]
        store.add_batch(records)
        assert store.frame_count() == 5

    def test_query_returns_results(self, store):
        store.add_batch(
            [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "timestamp_str": "00:00:00",
                    "description": "A dramatic car crash on a busy intersection.",
                },
                {
                    "frame_index": 1,
                    "timestamp_seconds": 10.0,
                    "timestamp_str": "00:00:10",
                    "description": "A beautiful sunset over the ocean.",
                },
            ]
        )
        results = store.query("car crash", n_results=2)
        assert len(results) > 0
        assert "timestamp_str" in results[0]
        assert "description" in results[0]
        assert "distance" in results[0]

    def test_query_top_result_relevant(self, store):
        store.add_batch(
            [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "timestamp_str": "00:00:00",
                    "description": "A vehicle collision with sparks flying.",
                },
                {
                    "frame_index": 1,
                    "timestamp_seconds": 60.0,
                    "timestamp_str": "00:01:00",
                    "description": "A person reading a book indoors.",
                },
            ]
        )
        results = store.query("car accident collision", n_results=2)
        # The crash frame should rank above the book frame
        assert results[0]["frame_index"] == 0

    def test_empty_store_query_returns_empty(self, store):
        results = store.query("anything", n_results=5)
        assert results == []

    def test_upsert_idempotent(self, store):
        store.add_frame_description(0, 0.0, "00:00:00", "First description.")
        store.add_frame_description(0, 0.0, "00:00:00", "Updated description.")
        assert store.frame_count() == 1
