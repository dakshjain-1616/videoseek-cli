"""
04_full_pipeline.py — End-to-end VideoSeek-CLI pipeline.

Demonstrates the complete workflow:
  1. Index a video (mock mode — no API key required)
  2. Run multiple natural-language queries
  3. Export matched frames to disk (writes text stubs in mock mode)
  4. Generate a Markdown + JSON report
  5. Browse the full frame timeline
  6. Print a summary table

Run:
    python examples/04_full_pipeline.py
"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import tempfile

# Setup: temp directories, mock mode
_tmp = tempfile.mkdtemp(prefix="vs_pipeline_")
os.environ.setdefault("OPENROUTER_API_KEY", "")
os.environ["VIDEOSEEK_DB_PATH"] = os.path.join(_tmp, "db")
os.environ["VIDEOSEEK_SIMILARITY_THRESHOLD"] = "0.05"
os.environ["VIDEOSEEK_EXPORT_DIR"] = os.path.join(_tmp, "exports")

# Use the included sample video, or override with VIDEO_PATH env var
_SAMPLE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "samples", "sample_video.mp4")
VIDEO = os.getenv("VIDEO_PATH", _SAMPLE)

# ── Step 1: Index ──────────────────────────────────────────────────────────────
from videoseek_cli import VideoAnalyzer, generate_json_report, generate_markdown_report, write_report

print("Step 1 — Indexing video …")
analyzer = VideoAnalyzer(VIDEO)
frame_count = analyzer.index(progress_callback=lambda done, total: None)
info = analyzer.get_video_info()
print(f"  Indexed {frame_count} frames | "
      f"Duration: {info['duration_str']} | "
      f"Resolution: {info['width']}x{info['height']}")

# ── Step 2: Run multiple queries ───────────────────────────────────────────────
print("\nStep 2 — Running queries …")
queries = [
    "Find the car crash",
    "Show the sunset",
    "Find a non-existent event: underwater volcano",
]
results_map = analyzer.batch_query(queries, top_k=3, threshold=0.05)
for q, hits in results_map.items():
    if hits:
        best = hits[0]
        print(f"  '{q}' -> {best.timestamp_str} (score={best.similarity_score:.3f})")
    else:
        print(f"  '{q}' -> no match")

# ── Step 3: Export matched frames ──────────────────────────────────────────────
print("\nStep 3 — Exporting frames …")
all_results = [r for hits in results_map.values() for r in hits[:1]]
exported = analyzer.export_frames(all_results, prefix="example")
for path in exported:
    print(f"  Wrote: {os.path.basename(path)}")

# ── Step 4: Generate reports ───────────────────────────────────────────────────
print("\nStep 4 — Generating reports …")
stats = analyzer.get_stats()

md_path = os.path.join(_tmp, "report.md")
write_report(VIDEO, results_map, md_path, fmt="markdown", stats=stats, video_info=info)
print(f"  Markdown : {md_path}")

json_path = os.path.join(_tmp, "report.json")
write_report(VIDEO, results_map, json_path, fmt="json", stats=stats, video_info=info)
print(f"  JSON     : {json_path}")

# ── Step 5: Browse the timeline ────────────────────────────────────────────────
print("\nStep 5 — Timeline (first 5 frames) …")
timeline = analyzer.get_timeline()
for frame in timeline[:5]:
    print(f"  [{frame['timestamp_str']}] {frame['description'][:70]}")
if len(timeline) > 5:
    print(f"  … and {len(timeline) - 5} more frames")

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n── Summary ────────────────────────────────────────────────────────────")
print(f"  Frames indexed : {stats['frames_indexed']}")
print(f"  Index time     : {stats['index_time_seconds']:.2f}s")
print(f"  Query time     : {stats['query_time_seconds']:.2f}s")
print(f"  Exported frames: {len(exported)}")
print(f"  Reports written: {md_path}, {json_path}")
print("\nDone.")
