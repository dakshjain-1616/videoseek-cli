"""
03_custom_config.py — Customising VideoSeek-CLI via environment variables.

Demonstrates:
  - Switching vision models (budget vs. premium)
  - Tuning FPS sampling rate
  - Adjusting similarity threshold
  - Changing the ChromaDB storage path
  - Inspecting the active config at runtime

Run:
    python examples/03_custom_config.py

You can also set these via a .env file in the project root.
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile

# --- Configuration via environment variables ---

# Vision model: use a budget model for fast/economical analysis
os.environ["VIDEOSEEK_MODEL"] = "google/gemini-2.5-flash"

# Sample 2 frames per second instead of the default 1
os.environ["VIDEOSEEK_FPS"] = "2"

# Cap at 50 frames so this example runs quickly
os.environ["VIDEOSEEK_MAX_FRAMES"] = "50"

# Require at least 20% similarity before showing a result
os.environ["VIDEOSEEK_SIMILARITY_THRESHOLD"] = "0.20"

# Store the index in a dedicated folder
os.environ["VIDEOSEEK_DB_PATH"] = tempfile.mkdtemp(prefix="vs_cfg_")

# No real API key → mock/demo mode
os.environ.setdefault("OPENROUTER_API_KEY", "")

# --- Inspect the active config ---
from videoseek_cli.config import (
    get_fps,
    get_max_frames,
    get_model,
    get_similarity_threshold,
    is_mock_mode,
)

print("Active configuration:")
print(f"  Model               : {get_model()}")
print(f"  FPS                 : {get_fps()}")
print(f"  Max frames          : {get_max_frames()}")
print(f"  Similarity threshold: {get_similarity_threshold()}")
print(f"  Mock mode           : {is_mock_mode()}")
print()

# --- Run a query with the custom settings ---
from videoseek_cli import VideoAnalyzer

# Use the included sample video, or override with VIDEO_PATH env var
_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples", "sample_video.mp4")
VIDEO = os.getenv("VIDEO_PATH", _SAMPLE)

analyzer = VideoAnalyzer(
    VIDEO,
    sample_fps=get_fps(),
    max_frames=get_max_frames(),
)
count = analyzer.index()
print(f"Indexed {count} frames at {get_fps()} fps.")

result = analyzer.query_best("Find the car crash")
if result:
    print(f"Best match: {result.timestamp_str}  (score={result.similarity_score:.3f})")
else:
    print("No match found above threshold.")
