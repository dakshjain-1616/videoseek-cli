"""Tests for frame extraction utilities."""

from __future__ import annotations

import pytest

from videoseek_cli.frame_extractor import (
    VideoFrame,
    extract_frames,
    frame_to_b64,
    get_video_info,
    seconds_to_timestamp,
)


class TestSecondsToTimestamp:
    def test_zero(self):
        assert seconds_to_timestamp(0) == "00:00:00"

    def test_seconds_only(self):
        assert seconds_to_timestamp(45) == "00:00:45"

    def test_minutes_and_seconds(self):
        assert seconds_to_timestamp(90) == "00:01:30"

    def test_hours_minutes_seconds(self):
        assert seconds_to_timestamp(3723) == "01:02:03"

    def test_large_value(self):
        ts = seconds_to_timestamp(83 * 60 + 45)  # 01:23:45
        assert ts == "01:23:45"

    def test_another_value(self):
        ts = seconds_to_timestamp(45 * 60 + 30)  # 00:45:30
        assert ts == "00:45:30"

    def test_rounding(self):
        # Should round to nearest second
        assert seconds_to_timestamp(1.6) == "00:00:02"


class TestGetVideoInfo:
    def test_returns_expected_keys(self, mock_video_path):
        info = get_video_info(mock_video_path)
        expected_keys = {
            "total_frames", "native_fps", "width", "height",
            "duration_seconds", "duration_str",
        }
        assert expected_keys.issubset(set(info.keys()))

    def test_dimensions(self, mock_video_path):
        info = get_video_info(mock_video_path)
        assert info["width"] == 320
        assert info["height"] == 240

    def test_fps_reasonable(self, mock_video_path):
        info = get_video_info(mock_video_path)
        assert 20 <= info["native_fps"] <= 30

    def test_invalid_path_raises(self):
        with pytest.raises(FileNotFoundError):
            get_video_info("/nonexistent/path/to/video.mp4")


class TestExtractFrames:
    def test_yields_video_frames(self, mock_video_path):
        frames = list(extract_frames(mock_video_path, sample_fps=1, max_frames=5))
        assert len(frames) > 0
        assert all(isinstance(f, VideoFrame) for f in frames)

    def test_max_frames_respected(self, mock_video_path):
        frames = list(extract_frames(mock_video_path, sample_fps=25, max_frames=3))
        assert len(frames) <= 3

    def test_frame_has_base64_image(self, mock_video_path):
        frames = list(extract_frames(mock_video_path, sample_fps=1, max_frames=1))
        assert len(frames) == 1
        f = frames[0]
        assert isinstance(f.image_b64, str)
        assert len(f.image_b64) > 100  # Non-empty base64

    def test_frame_timestamp_increases(self, mock_video_path):
        frames = list(extract_frames(mock_video_path, sample_fps=1, max_frames=5))
        timestamps = [f.timestamp_seconds for f in frames]
        assert timestamps == sorted(timestamps)

    def test_frame_timestamp_str_format(self, mock_video_path):
        frames = list(extract_frames(mock_video_path, sample_fps=1, max_frames=2))
        for f in frames:
            parts = f.timestamp_str.split(":")
            assert len(parts) == 3
            assert all(p.isdigit() for p in parts)
