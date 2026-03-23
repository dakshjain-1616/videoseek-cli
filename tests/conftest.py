"""Shared fixtures for VideoSeek-CLI tests."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

# Force mock mode for all tests — no real API key or video file required
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ.setdefault("VIDEOSEEK_DB_PATH", tempfile.mkdtemp(prefix="vs_test_db_"))
os.environ.setdefault("VIDEOSEEK_SIMILARITY_THRESHOLD", "0.1")


@pytest.fixture(scope="session")
def mock_video_path(tmp_path_factory) -> str:
    """Create a minimal synthetic MP4 video using OpenCV for use in tests."""
    import cv2

    tmp_dir = tmp_path_factory.mktemp("videos")
    video_path = str(tmp_dir / "video.mp4")

    width, height, fps, total_frames = 320, 240, 25, 150  # 6-second video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    rng = np.random.default_rng(42)
    for i in range(total_frames):
        # Vary the frame slightly to simulate a real video
        frame = rng.integers(0, 255, (height, width, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()

    assert os.path.exists(video_path), "Failed to create synthetic test video"
    return video_path


@pytest.fixture
def fresh_db_path(tmp_path) -> str:
    """Return a temporary ChromaDB path, isolated per test."""
    db = str(tmp_path / "test_db")
    os.environ["VIDEOSEEK_DB_PATH"] = db
    return db
