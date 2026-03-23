# VideoSeek-CLI Examples

Runnable Python scripts demonstrating VideoSeek-CLI features from simple to
advanced. All examples work in **mock mode** (no API key required).

```bash
# Run any example from the project root:
python examples/01_quick_start.py
```

---

## Scripts

| Script | What it demonstrates |
|--------|---------------------|
| [`01_quick_start.py`](01_quick_start.py) | Minimal working example — create an analyzer, index a video, run one query, print the best matching timestamp |
| [`02_advanced_usage.py`](02_advanced_usage.py) | Top-k results, custom threshold, batch querying multiple topics in one pass, runtime statistics, force re-indexing |
| [`03_custom_config.py`](03_custom_config.py) | Configure vision model, FPS sampling rate, similarity threshold, and storage path via environment variables |
| [`04_full_pipeline.py`](04_full_pipeline.py) | End-to-end workflow: index → batch query → export frames → generate Markdown & JSON reports → browse timeline |

---

## Requirements

Install project dependencies first:

```bash
pip install -r requirements.txt
```

## Mock mode vs. live mode

All examples call `os.environ.setdefault("OPENROUTER_API_KEY", "")` which
triggers **mock mode** when no real key is present. In mock mode:

- No real video file is needed — the analyzer generates synthetic frame data.
- No API calls are made — descriptions are produced locally.
- All features (queries, exports, reports) still run end-to-end.

To switch to **live mode**, set your key before running:

```bash
export OPENROUTER_API_KEY=sk-or-...
python examples/04_full_pipeline.py
```
