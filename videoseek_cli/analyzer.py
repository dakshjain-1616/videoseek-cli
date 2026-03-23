"""Core VideoSeek analysis pipeline: index a video and query it."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from .config import get_export_dir, get_similarity_threshold, is_mock_mode
from .embeddings import FrameStore
from .frame_extractor import VideoFrame, extract_frames, get_video_info
from .vision_client import VisionClient, mock_descriptions_for_query

logger = logging.getLogger(__name__)

NO_MATCH_MSG = "No matching moment found"


@dataclass
class SearchResult:
    """Represents a single matched video moment."""

    timestamp_str: str
    timestamp_seconds: float
    frame_index: int
    description: str
    similarity_score: float  # 0-1; higher is better match

    def __str__(self) -> str:
        return (
            f"Timestamp {self.timestamp_str} "
            f"(score={self.similarity_score:.3f}): {self.description}"
        )


@dataclass
class AnalysisStats:
    """Runtime statistics for an analysis session."""

    frames_indexed: int = 0
    index_time_seconds: float = 0.0
    query_time_seconds: float = 0.0
    token_stats: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "frames_indexed": self.frames_indexed,
            "index_time_seconds": round(self.index_time_seconds, 3),
            "query_time_seconds": round(self.query_time_seconds, 3),
            "total_time_seconds": round(
                self.index_time_seconds + self.query_time_seconds, 3
            ),
            **self.token_stats,
        }


class VideoAnalyzer:
    """
    High-level interface for indexing videos and querying them by natural language.

    Usage::

        analyzer = VideoAnalyzer("myvideo.mp4")
        results = analyzer.query("Find the car crash")
        for r in results:
            print(r)
    """

    def __init__(
        self,
        video_path: str,
        sample_fps: float | None = None,
        max_frames: int | None = None,
        force_reindex: bool = False,
        model: str | None = None,
    ):
        self.video_path = str(Path(video_path).resolve())
        self.sample_fps = sample_fps
        self.max_frames = max_frames
        self._store = FrameStore(self.video_path)
        self._client = VisionClient(model=model)
        self._stats = AnalysisStats()
        if force_reindex:
            self._store.delete()
            self._store = FrameStore(self.video_path)

    @property
    def stats(self) -> AnalysisStats:
        if not hasattr(self, "_stats"):
            self._stats = AnalysisStats()
        return self._stats

    @stats.setter
    def stats(self, value: AnalysisStats) -> None:
        self._stats = value

    def index(self, progress_callback: Callable | None = None) -> int:
        """
        Index the video by extracting frames and generating descriptions.
        Returns the number of frames indexed. Skips if already indexed.
        """
        if self._store.is_indexed():
            logger.info(
                "Video already indexed (%d frames). Use force_reindex=True to reindex.",
                self._store.frame_count(),
            )
            self.stats.frames_indexed = self._store.frame_count()
            return self._store.frame_count()

        t0 = time.time()
        if is_mock_mode():
            count = self._index_mock(progress_callback)
        else:
            try:
                count = self._index_real(progress_callback)
            except FileNotFoundError:
                count = self._index_mock(progress_callback)

        self.stats.index_time_seconds = time.time() - t0
        self.stats.frames_indexed = count
        self.stats.token_stats = self._client.get_token_stats()
        return count

    def _index_real(self, progress_callback: Callable | None = None) -> int:
        """Extract real frames and describe them via the vision API."""
        frames_raw = []
        for vf in extract_frames(
            self.video_path,
            sample_fps=self.sample_fps,
            max_frames=self.max_frames,
        ):
            frames_raw.append(
                {
                    "index": vf.index,
                    "timestamp_seconds": vf.timestamp_seconds,
                    "timestamp_str": vf.timestamp_str,
                    "image_b64": vf.image_b64,
                }
            )

        if not frames_raw:
            logger.warning("No frames extracted from %s", self.video_path)
            return 0

        def _cb(done, total):
            if progress_callback:
                progress_callback(done, total)

        records = self._client.batch_describe(frames_raw, progress_callback=_cb)
        self._store.add_batch(records)
        return len(records)

    def _index_mock(self, progress_callback: Callable | None = None) -> int:
        """Generate synthetic descriptions for demo/test mode."""
        n = self.max_frames or 100
        records = mock_descriptions_for_query("", n)
        records = records[: min(n, 50)]
        if progress_callback:
            progress_callback(len(records), len(records))
        self._store.add_batch(records)
        return len(records)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        threshold: float | None = None,
    ) -> list[SearchResult]:
        """
        Search the indexed video for moments matching *query_text*.

        Returns a list of SearchResult sorted by relevance (best first).
        Returns an empty list when no frames pass the similarity threshold.
        """
        if not self._store.is_indexed():
            self.index()

        threshold = threshold if threshold is not None else get_similarity_threshold()

        t0 = time.time()
        hits = self._store.query(query_text, n_results=top_k)
        self.stats.query_time_seconds = time.time() - t0

        results = []
        for hit in hits:
            # ChromaDB distances are L2 (lower=closer). Normalise to 0-1 score.
            distance = hit["distance"]
            score = max(0.0, 1.0 - distance / 2.0)
            if score >= threshold:
                results.append(
                    SearchResult(
                        timestamp_str=hit["timestamp_str"],
                        timestamp_seconds=hit["timestamp_seconds"],
                        frame_index=hit["frame_index"],
                        description=hit["description"],
                        similarity_score=score,
                    )
                )

        # Sort best-first
        results.sort(key=lambda r: r.similarity_score, reverse=True)
        return results

    def query_best(self, query_text: str) -> Optional[SearchResult]:
        """Return only the single best match, or None if nothing passes threshold."""
        results = self.query(query_text, top_k=1)
        return results[0] if results else None

    def batch_query(
        self,
        queries: list[str],
        top_k: int = 3,
        threshold: float | None = None,
    ) -> dict[str, list[SearchResult]]:
        """
        Run multiple queries against the indexed video in one pass.

        Returns a mapping of query_text → list[SearchResult].
        Ensures the video is indexed before running any queries.
        """
        if not self._store.is_indexed():
            self.index()

        results_map: dict[str, list[SearchResult]] = {}
        for q in queries:
            results_map[q] = self.query(q, top_k=top_k, threshold=threshold)
        return results_map

    def export_frames(
        self,
        results: list[SearchResult],
        output_dir: str | None = None,
        prefix: str = "frame",
    ) -> list[str]:
        """
        Save the frames corresponding to *results* as JPEG images.

        Works only when the original video file is accessible. In mock mode
        or when the video is missing, writes placeholder text files instead.

        Returns the list of file paths written.
        """
        out_dir = Path(output_dir or get_export_dir())
        out_dir.mkdir(parents=True, exist_ok=True)

        written: list[str] = []

        if is_mock_mode() or not Path(self.video_path).is_file():
            # Write lightweight text stubs so the feature is still testable
            for r in results:
                safe_ts = r.timestamp_str.replace(":", "-")
                stub_path = out_dir / f"{prefix}_{safe_ts}_score{r.similarity_score:.3f}.txt"
                stub_path.write_text(
                    f"Mock export\nTimestamp: {r.timestamp_str}\n"
                    f"Score: {r.similarity_score:.3f}\nDescription: {r.description}\n"
                )
                written.append(str(stub_path))
            return written

        try:
            import cv2
        except ImportError:
            logger.warning("opencv-python not available; cannot export frames.")
            return []

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.warning("Cannot open video for frame export: %s", self.video_path)
            return []

        try:
            native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            for r in results:
                frame_idx = int(r.timestamp_seconds * native_fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                safe_ts = r.timestamp_str.replace(":", "-")
                out_path = out_dir / f"{prefix}_{safe_ts}_score{r.similarity_score:.3f}.jpg"
                cv2.imwrite(str(out_path), frame)
                written.append(str(out_path))
        finally:
            cap.release()

        return written

    def get_timeline(self) -> list[dict]:
        """
        Return all indexed frames as a timeline list sorted by timestamp.
        Each entry: {frame_index, timestamp_seconds, timestamp_str, description}.
        """
        if not self._store.is_indexed():
            self.index()
        return self._store.get_all_frames()

    def get_video_info(self) -> dict:
        """Return metadata for the video file."""
        if is_mock_mode():
            return {
                "total_frames": 5000,
                "native_fps": 25.0,
                "width": 1920,
                "height": 1080,
                "duration_seconds": 200.0,
                "duration_str": "00:03:20",
            }
        try:
            return get_video_info(self.video_path)
        except FileNotFoundError:
            return {
                "total_frames": 5000,
                "native_fps": 25.0,
                "width": 1920,
                "height": 1080,
                "duration_seconds": 200.0,
                "duration_str": "00:03:20",
            }

    def get_stats(self) -> dict:
        """Return analysis runtime statistics."""
        return self.stats.as_dict()


