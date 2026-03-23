"""Extract frames from video files at a configurable sampling rate."""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Generator

import cv2
import numpy as np
from PIL import Image

from .config import get_fps, get_frame_quality, get_frame_size, get_max_frames, is_debug

logger = logging.getLogger(__name__)


@dataclass
class VideoFrame:
    """A single extracted video frame with metadata."""

    index: int
    timestamp_seconds: float
    timestamp_str: str
    image_b64: str  # base64-encoded JPEG
    width: int
    height: int


def seconds_to_timestamp(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def frame_to_b64(frame_bgr: np.ndarray, quality: int | None = None, size: int | None = None) -> str:
    """Convert an OpenCV BGR frame to a base64-encoded JPEG string."""
    q = quality if quality is not None else get_frame_quality()
    s = size if size is not None else get_frame_size()
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    pil_img.thumbnail((s, s), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / native_fps if native_fps > 0 else 0.0
        return {
            "total_frames": total_frames,
            "native_fps": native_fps,
            "width": width,
            "height": height,
            "duration_seconds": duration,
            "duration_str": seconds_to_timestamp(duration),
        }
    finally:
        cap.release()


def extract_frames(
    video_path: str,
    sample_fps: float | None = None,
    max_frames: int | None = None,
) -> Generator[VideoFrame, None, None]:
    """
    Yield VideoFrame objects sampled from *video_path*.

    Args:
        video_path: Path to the video file.
        sample_fps: How many frames to sample per second.  Defaults to
                    VIDEOSEEK_FPS env var (1 fps).
        max_frames: Hard cap on total frames yielded.  Defaults to
                    VIDEOSEEK_MAX_FRAMES env var (300).
    """
    sample_fps = sample_fps if sample_fps is not None else get_fps()
    max_frames = max_frames if max_frames is not None else get_max_frames()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # How many native frames to skip between each sampled frame
        step = max(1, int(native_fps / sample_fps))
        frame_indices = list(range(0, total_frames, step))[:max_frames]

        if is_debug():
            logger.debug(
                "Video: %s | native_fps=%.2f total_frames=%d step=%d samples=%d",
                Path(video_path).name,
                native_fps,
                total_frames,
                step,
                len(frame_indices),
            )

        for idx, frame_idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            ts = frame_idx / native_fps
            yield VideoFrame(
                index=idx,
                timestamp_seconds=ts,
                timestamp_str=seconds_to_timestamp(ts),
                image_b64=frame_to_b64(frame),
                width=width,
                height=height,
            )
    finally:
        cap.release()
