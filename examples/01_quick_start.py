"""
01_quick_start.py — Minimal working example of VideoSeek-CLI.

Demonstrates:
  - Creating a VideoAnalyzer
  - Indexing a video (mock mode when no API key is set)
  - Running a single natural-language query
  - Printing the best matching timestamp

Run:
    python examples/01_quick_start.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile

# Use a temp DB so the example is always clean
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ["VIDEOSEEK_DB_PATH"] = tempfile.mkdtemp(prefix="vs_qs_")
os.environ["VIDEOSEEK_SIMILARITY_THRESHOLD"] = "0.1"

from videoseek_cli import NO_MATCH_MSG, VideoAnalyzer

# Use the included sample video, or override with VIDEO_PATH env var
_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples", "sample_video.mp4")
VIDEO = os.getenv("VIDEO_PATH", _SAMPLE)

analyzer = VideoAnalyzer(VIDEO)

# Index the video (builds the semantic frame store)
frame_count = analyzer.index()
print(f"Indexed {frame_count} frames.")

# Query with plain English
result = analyzer.query_best("Find the car crash")

if result:
    print(f"Best match: {result.timestamp_str}  (score={result.similarity_score:.3f})")
    print(f"Description: {result.description}")
else:
    print(NO_MATCH_MSG)
