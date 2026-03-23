"""
02_advanced_usage.py — Advanced VideoSeek-CLI features.

Demonstrates:
  - Top-k results (not just best match)
  - Custom similarity threshold
  - Batch querying multiple topics in one pass
  - Accessing runtime statistics
  - Forcing re-indexing

Run:
    python examples/02_advanced_usage.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile

os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ["VIDEOSEEK_DB_PATH"] = tempfile.mkdtemp(prefix="vs_adv_")
os.environ["VIDEOSEEK_SIMILARITY_THRESHOLD"] = "0.05"

from videoseek_cli import VideoAnalyzer

# Use the included sample video, or override with VIDEO_PATH env var
_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples", "sample_video.mp4")
VIDEO = os.getenv("VIDEO_PATH", _SAMPLE)

# --- 1. Top-k results with a custom threshold ---
print("=== Top-3 results ===")
analyzer = VideoAnalyzer(VIDEO)
analyzer.index()

results = analyzer.query("Find the sunset", top_k=3, threshold=0.05)
for i, r in enumerate(results, 1):
    print(f"  [{i}] {r.timestamp_str}  score={r.similarity_score:.3f}  {r.description[:60]}")

# --- 2. Batch query — index once, query many ---
print("\n=== Batch query ===")
queries = ["car crash", "sunset over the ocean", "people walking"]
batch = analyzer.batch_query(queries, top_k=2, threshold=0.05)
for q, hits in batch.items():
    top = hits[0].timestamp_str if hits else "no match"
    print(f"  '{q}' -> {top}")

# --- 3. Runtime statistics ---
print("\n=== Stats ===")
stats = analyzer.get_stats()
for k, v in stats.items():
    print(f"  {k}: {v}")

# --- 4. Force re-index (discard cached data and re-analyse) ---
print("\n=== Force re-index ===")
analyzer2 = VideoAnalyzer(VIDEO, force_reindex=True)
count = analyzer2.index()
print(f"  Re-indexed {count} frames.")
