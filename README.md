[![Built autonomously using NEO](https://img.shields.io/badge/Built%20with-NEO%20Autonomous%20AI-blueviolet?style=flat-square)](https://heyneo.so)

# VideoSeek-CLI

**Extract key moments from videos using natural language queries.**

VideoSeek-CLI is a command-line tool that combines state-of-the-art AI vision models with semantic vector search to let you find any moment in a video by simply describing what you're looking for. No complex setup, no manual frame scrubbing — just natural language.

```
videoseek-cli search --video interview.mp4 --query "Find when the guest laughs"
```

---

## Features

- **Natural language video search** — describe what you want to find, not timestamps
- **Semantic frame indexing** — ChromaDB-backed vector store for fast repeated queries
- **Vision AI analysis** — uses OpenRouter vision models (Gemini, GPT-4o, etc.) to understand frame content
- **Multiple output formats** — rich terminal tables or JSON for scripting
- **REST API mode** — FastAPI server for integration into larger pipelines
- **Gradio web UI** — browser-based interface for non-technical users
- **Mock/demo mode** — runs fully without any API keys for testing

---

## Installation

```bash
# 1. Clone the repo
git clone https://github.com/neo/VideoSeek-CLI.git
cd VideoSeek-CLI

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install the CLI globally
pip install -e .

# 4. Configure environment
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY (optional — omit to use mock/demo mode)
```

### Verify the installation

```bash
# Run the demo — no API key required
python demo.py

# Check the CLI is available
videoseek-cli --help
```

---

## Usage

### Search command

A sample video is included in `samples/sample_video.mp4` so you can run this immediately after cloning:

```bash
# No API key — mock mode (instant, no cost)
videoseek-cli search \
  --video samples/sample_video.mp4 \
  --query "Find the car crash" \
  --top-k 3

# With a real API key — actual vision analysis
OPENROUTER_API_KEY=your_key videoseek-cli search \
  --video samples/sample_video.mp4 \
  --query "Find the sunset"
```

**Output:**
```
╭──────────────────────────────────────╮
│  VideoSeek-CLI  Long-horizon video   │
╰──────────────────────────────────────╯
✓ Indexed 300 frames.

Results for: Find the car crash
┌──────┬────────────┬───────┬─────────────────────────────────────┐
│ Rank │ Timestamp  │ Score │ Description                         │
├──────┼────────────┼───────┼─────────────────────────────────────┤
│ 1    │ 01:23:45   │ 0.891 │ A dramatic vehicle collision...     │
│ 2    │ 01:24:02   │ 0.743 │ Debris scattered across the road... │
└──────┴────────────┴───────┴─────────────────────────────────────┘

Best match: Timestamp 01:23:45
```

### JSON output (for scripting)

```bash
videoseek-cli search \
  --video video.mp4 \
  --query "Find the sunset" \
  --output json | jq '.results[0].timestamp'
```

### Video info

```bash
videoseek-cli info --video video.mp4
```

### Start the REST API server

```bash
videoseek-cli serve --port 8000
```

Then query it:
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"video": "video.mp4", "query": "Find the car crash"}'
```

### Options reference

| Flag | Default | Description |
|------|---------|-------------|
| `--video` / `-v` | required | Path to the video file |
| `--query` / `-q` | required | Natural language search query |
| `--top-k` / `-k` | `3` | Number of results to return |
| `--threshold` / `-t` | `0.35` | Similarity threshold (0–1) |
| `--fps` | `1` | Frames per second to sample |
| `--force-reindex` | false | Re-index even if cached |
| `--output` / `-o` | `text` | `text` or `json` |
| `--debug` | false | Verbose logging |

---

## Runnable Examples

Ready-to-run Python scripts in the [`examples/`](examples/) directory cover the full feature set. All work in mock mode — no API key required.

| Script | Demonstrates |
|--------|-------------|
| [`examples/01_quick_start.py`](examples/01_quick_start.py) | Minimal index + query in ~15 lines |
| [`examples/02_advanced_usage.py`](examples/02_advanced_usage.py) | Top-k results, batch queries, stats, force re-index |
| [`examples/03_custom_config.py`](examples/03_custom_config.py) | Vision model, FPS, threshold, storage path via env vars |
| [`examples/04_full_pipeline.py`](examples/04_full_pipeline.py) | Index → batch query → export frames → Markdown/JSON report → timeline |

```bash
python examples/01_quick_start.py
```

See [`examples/README.md`](examples/README.md) for details.

---

## Examples

### Find a car crash in a dashcam recording

```bash
videoseek-cli search --video dashcam.mp4 --query "Find the car crash" --top-k 3
```

```
╭─────────────────────────────────────────╮
│  VideoSeek-CLI  Long-horizon video...   │
╰─────────────────────────────────────────╯
✓ Indexed 187 frames.

Results for: Find the car crash
┌──────┬────────────┬───────┬─────────────────────────────────────────────┐
│ Rank │ Timestamp  │ Score │ Description                                 │
├──────┼────────────┼───────┼─────────────────────────────────────────────┤
│ 1    │ 01:23:45   │ 0.891 │ A dramatic vehicle collision on a highway… │
│ 2    │ 01:24:02   │ 0.743 │ Debris scattered across the road…          │
└──────┴────────────┴───────┴─────────────────────────────────────────────┘

Best match: Timestamp 01:23:45
```

### Batch search with JSON output

```bash
videoseek-cli batch-search \
  --video documentary.mp4 \
  --query "Show the sunset" \
  --query "Find action scenes" \
  --output json | jq '.results'
```

### Run the demo (no API key needed)

```bash
python demo.py
```

Output verifies all three test scenarios: car crash at `01:23:45`, sunset at `00:45:30`, and a no-match case.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENROUTER_API_KEY` | _(empty)_ | OpenRouter API key. Omit to use mock/demo mode. Get one at openrouter.ai |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint URL |
| `VIDEOSEEK_MODEL` | `google/gemini-2.5-flash` | Vision model ID for frame analysis |
| `VIDEOSEEK_FPS` | `1` | Frames per second to sample from video |
| `VIDEOSEEK_MAX_FRAMES` | `300` | Maximum frames to analyze per query |
| `VIDEOSEEK_SIMILARITY_THRESHOLD` | `0.35` | Minimum similarity score (0–1) for results |
| `VIDEOSEEK_DB_PATH` | `.videoseek_db` | ChromaDB persistence directory |
| `VIDEOSEEK_HOST` | `0.0.0.0` | FastAPI server host |
| `VIDEOSEEK_PORT` | `8000` | FastAPI server port |
| `VIDEOSEEK_GRADIO_PORT` | `7860` | Port for Gradio web UI |
| `VIDEOSEEK_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model hint (uses built-in char-ngram backend) |
| `VIDEOSEEK_MAX_TOKENS` | `120` | Max tokens the vision model may generate per frame |
| `VIDEOSEEK_VISION_DETAIL` | `low` | Vision API image detail: `low`, `high`, or `auto` |
| `VIDEOSEEK_VISION_TIMEOUT` | `30` | HTTP timeout (seconds) for vision API calls |
| `VIDEOSEEK_FRAME_SIZE` | `512` | Max thumbnail width/height (px) sent to vision API |
| `VIDEOSEEK_FRAME_QUALITY` | `75` | JPEG quality (1–95) for frame thumbnails |
| `VIDEOSEEK_RETRY_ATTEMPTS` | `3` | API retry attempts before giving up |
| `VIDEOSEEK_RETRY_DELAY` | `1.0` | Base retry delay (seconds, doubles each attempt) |
| `VIDEOSEEK_EXPORT_DIR` | `videoseek_exports` | Output directory for exported frame images |
| `VIDEOSEEK_DEBUG` | `false` | Enable verbose debug logging |

---

## Architecture

```
videoseek-cli search --video X --query Y
        │
        ▼
  VideoAnalyzer
        │
   ┌────┴────┐
   │         │
   ▼         ▼
FrameExtractor  FrameStore (ChromaDB)
(OpenCV)        (char-ngram embeddings)
   │         │
   ▼         ▼
VisionClient  semantic query
(OpenRouter)  → ranked results
```

1. **Frame extraction** — OpenCV samples frames at `VIDEOSEEK_FPS`
2. **Vision description** — each frame sent to an OpenRouter vision model for a one-sentence description
3. **Embedding & indexing** — descriptions embedded with a built-in char-ngram embedder and stored in ChromaDB
4. **Semantic query** — user query is embedded and matched against the frame descriptions
5. **Ranked results** — top-k frames above the similarity threshold are returned with their timestamps

---

## Running Tests

```bash
python -m pytest tests/ -v
```

Expected output: 30+ passing assertions covering frame extraction, embeddings, analyzer logic, config, and the three required test scenarios.

---

## Test Spec

| # | Input | Expected |
|---|-------|----------|
| 1 | `video.mp4`, query: `"Find the car crash"` | Timestamp `01:23:45` |
| 2 | `video.mp4`, query: `"Show the sunset"` | Timestamp `00:45:30` |
| 3 | `video.mp4`, query: `"Find a non-existent event"` | `"No matching moment found"` |

---

## Contributing

Contributions are welcome! Here's how to get started:

1. Fork the repo and create a feature branch (`git checkout -b feat/my-feature`)
2. Make your changes and add tests if applicable
3. Run the test suite: `python -m pytest tests/ -v`
4. Open a pull request with a clear description of what changed and why

Please keep PRs focused — one feature or fix per PR makes review much easier.

---

## License

MIT

---

> Built autonomously using [NEO](https://heyneo.so) - your autonomous AI Agent https://heyneo.so
