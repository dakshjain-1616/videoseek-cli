"""Gradio web UI for VideoSeek-CLI — tabbed interface with live stats."""

from __future__ import annotations

import time
from pathlib import Path

import gradio as gr

from .analyzer import NO_MATCH_MSG, VideoAnalyzer
from .config import (
    ALL_MODELS,
    BUDGET_MODELS,
    PREMIUM_MODELS,
    get_gradio_port,
    get_model,
    is_mock_mode,
)

# ---------------------------------------------------------------------------
# Scenario cards — example queries shown in the UI
# ---------------------------------------------------------------------------

_SCENARIOS = [
    "Find the car crash scene",
    "Show the beautiful sunset",
    "Find the most intense action scene",
    "Show scenes with people talking",
    "Find any fire or explosion",
    "Show water or ocean scenes",
    "Find busy city street shots",
    "Show athletic or sports moments",
]

_MOCK_NOTE = (
    "\n> ⚠️ **Mock/demo mode** — set `OPENROUTER_API_KEY` for real analysis."
    if is_mock_mode()
    else ""
)

_HEADER_MD = (
    "# 🎬 VideoSeek-CLI\n"
    "**Find any moment in a video using natural language** — powered by AI vision + semantic search.\n\n"
    "_Built autonomously by [NEO](https://heyneo.so) — your autonomous AI Agent_"
    + _MOCK_NOTE
)


# ---------------------------------------------------------------------------
# Core handler functions (called by Gradio event listeners)
# ---------------------------------------------------------------------------

def _do_search(
    video_path: str,
    query: str,
    top_k: int,
    threshold: float,
    model: str,
) -> tuple[str, str]:
    """Run single-query search; returns (results_markdown, stats_markdown)."""
    if not video_path:
        return "Please upload a video file.", ""
    if not query.strip():
        return "Please enter a query.", ""

    t0 = time.time()
    analyzer = VideoAnalyzer(video_path=video_path, model=model or None)
    analyzer.index()
    results = analyzer.query(query, top_k=int(top_k), threshold=float(threshold))
    elapsed = time.time() - t0

    stats = analyzer.get_stats()
    stats_md = _format_stats(stats, elapsed)

    if not results:
        return NO_MATCH_MSG, stats_md

    lines = [f"**Results for:** _{query}_\n"]
    for i, r in enumerate(results, 1):
        bar = "█" * int(r.similarity_score * 20)
        lines.append(
            f"**{i}. Timestamp `{r.timestamp_str}`** — score `{r.similarity_score:.3f}` {bar}\n"
            f"> {r.description}\n"
        )
    lines.append(f"\n**Best match: `{results[0].timestamp_str}`**")
    return "\n".join(lines), stats_md


def _do_batch_search(
    video_path: str,
    queries_text: str,
    top_k: int,
    threshold: float,
    model: str,
) -> tuple[str, str]:
    """Run multiple queries; returns (results_markdown, stats_markdown)."""
    if not video_path:
        return "Please upload a video file.", ""
    queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
    if not queries:
        return "Please enter at least one query (one per line).", ""

    t0 = time.time()
    analyzer = VideoAnalyzer(video_path=video_path, model=model or None)
    analyzer.index()
    results_map = analyzer.batch_query(queries, top_k=int(top_k), threshold=float(threshold))
    elapsed = time.time() - t0

    stats = analyzer.get_stats()
    stats_md = _format_stats(stats, elapsed)

    lines = [f"**Batch search across {len(queries)} queries**\n---\n"]
    for q, rs in results_map.items():
        lines.append(f"### {q}")
        if not rs:
            lines.append(f"*{NO_MATCH_MSG}*\n")
        else:
            for i, r in enumerate(rs, 1):
                bar = "█" * int(r.similarity_score * 20)
                lines.append(
                    f"**{i}.** `{r.timestamp_str}` — `{r.similarity_score:.3f}` {bar}  \n"
                    f"> {r.description}\n"
                )
    return "\n".join(lines), stats_md


def _do_video_info(video_path: str) -> str:
    """Return a markdown table of video metadata for the given file."""
    if not video_path:
        return "Please upload a video file."
    analyzer = VideoAnalyzer(video_path=video_path)
    info = analyzer.get_video_info()
    lines = ["**Video Metadata**\n"]
    for k, v in info.items():
        label = k.replace("_", " ").title()
        lines.append(f"- **{label}:** {v}")
    return "\n".join(lines)


def _do_export(
    video_path: str,
    query: str,
    top_k: int,
    threshold: float,
    model: str,
    export_dir: str,
) -> str:
    """Search the video, then save matched frames as JPEG images to disk."""
    if not video_path:
        return "Please upload a video file."
    if not query.strip():
        return "Please enter a query."

    analyzer = VideoAnalyzer(video_path=video_path, model=model or None)
    analyzer.index()
    results = analyzer.query(query, top_k=int(top_k), threshold=float(threshold))
    if not results:
        return NO_MATCH_MSG

    exported = analyzer.export_frames(results, output_dir=export_dir or None)
    if not exported:
        return "No frames could be exported (video file may not be accessible)."
    lines = [f"**Exported {len(exported)} frame(s):**\n"]
    for p in exported:
        lines.append(f"- `{p}`")
    return "\n".join(lines)


def _format_stats(stats: dict, wall_time: float) -> str:
    """Format an analysis stats dict as a markdown table, including token usage when present."""
    if not stats:
        return ""
    lines = [
        "### Live Stats",
        "| Metric | Value |",
        "|---|---|",
        f"| Frames indexed | {stats.get('frames_indexed', '—')} |",
        f"| Index time | {stats.get('index_time_seconds', 0):.3f}s |",
        f"| Query time | {stats.get('query_time_seconds', 0):.3f}s |",
        f"| Total wall time | {wall_time:.2f}s |",
    ]
    if stats.get("total_tokens"):
        lines += [
            f"| Prompt tokens | {stats.get('prompt_tokens', 0)} |",
            f"| Completion tokens | {stats.get('completion_tokens', 0)} |",
            f"| Total tokens | {stats.get('total_tokens', 0)} |",
            f"| API calls | {stats.get('api_calls', 0)} |",
        ]
    if stats.get("retries"):
        lines.append(f"| Retries | {stats['retries']} |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    """Construct and return the Gradio Blocks UI for VideoSeek-CLI."""
    model_choices = PREMIUM_MODELS + BUDGET_MODELS
    default_model = get_model()

    with gr.Blocks(title="VideoSeek-CLI", theme=gr.themes.Soft()) as demo:

        gr.Markdown(_HEADER_MD)

        with gr.Tabs():

            # ------------------------------------------------------------------
            # Tab 1 — Single Search
            # ------------------------------------------------------------------
            with gr.TabItem("🔍 Search"):
                with gr.Row():
                    with gr.Column():
                        s_video = gr.Video(label="Upload Video")
                        s_query = gr.Textbox(
                            label="Query",
                            placeholder="e.g. Find the car crash",
                        )

                        with gr.Accordion("Example Queries", open=False):
                            s_scenario = gr.Radio(
                                choices=_SCENARIOS,
                                label="Pick an example query",
                                value=None,
                            )
                            s_scenario.change(
                                fn=lambda q: q or "",
                                inputs=s_scenario,
                                outputs=s_query,
                            )

                        with gr.Row():
                            s_top_k = gr.Slider(1, 10, step=1, value=3, label="Top K results")
                            s_threshold = gr.Slider(
                                0.0, 1.0, step=0.05, value=0.35, label="Similarity threshold"
                            )
                        s_model = gr.Dropdown(
                            choices=model_choices,
                            value=default_model,
                            label="Vision model",
                            info="OpenRouter model ID — premium = most accurate, budget = fastest",
                        )
                        s_btn = gr.Button("Search", variant="primary")

                    with gr.Column():
                        s_output = gr.Markdown(label="Results")
                        s_stats = gr.Markdown(label="Stats")

                s_btn.click(
                    fn=_do_search,
                    inputs=[s_video, s_query, s_top_k, s_threshold, s_model],
                    outputs=[s_output, s_stats],
                )

            # ------------------------------------------------------------------
            # Tab 2 — Batch Search
            # ------------------------------------------------------------------
            with gr.TabItem("📋 Batch Search"):
                with gr.Row():
                    with gr.Column():
                        b_video = gr.Video(label="Upload Video")
                        b_queries = gr.Textbox(
                            label="Queries (one per line)",
                            placeholder=(
                                "Find the car crash\nShow the sunset\nFind action scenes"
                            ),
                            lines=6,
                        )
                        with gr.Row():
                            b_top_k = gr.Slider(1, 10, step=1, value=3, label="Top K results")
                            b_threshold = gr.Slider(
                                0.0, 1.0, step=0.05, value=0.35, label="Similarity threshold"
                            )
                        b_model = gr.Dropdown(
                            choices=model_choices,
                            value=default_model,
                            label="Vision model",
                        )
                        b_btn = gr.Button("Run Batch Search", variant="primary")

                    with gr.Column():
                        b_output = gr.Markdown(label="Results")
                        b_stats = gr.Markdown(label="Stats")

                b_btn.click(
                    fn=_do_batch_search,
                    inputs=[b_video, b_queries, b_top_k, b_threshold, b_model],
                    outputs=[b_output, b_stats],
                )

            # ------------------------------------------------------------------
            # Tab 3 — Export Frames
            # ------------------------------------------------------------------
            with gr.TabItem("💾 Export Frames"):
                with gr.Row():
                    with gr.Column():
                        ex_video = gr.Video(label="Upload Video")
                        ex_query = gr.Textbox(
                            label="Query",
                            placeholder="e.g. Find the car crash",
                        )
                        with gr.Row():
                            ex_top_k = gr.Slider(1, 10, step=1, value=3, label="Top K results")
                            ex_threshold = gr.Slider(
                                0.0, 1.0, step=0.05, value=0.35, label="Similarity threshold"
                            )
                        ex_model = gr.Dropdown(
                            choices=model_choices,
                            value=default_model,
                            label="Vision model",
                        )
                        ex_dir = gr.Textbox(
                            label="Export Directory",
                            value="videoseek_exports",
                            placeholder="Output folder for frame images",
                        )
                        ex_btn = gr.Button("Export Frames", variant="primary")

                    with gr.Column():
                        ex_output = gr.Markdown(label="Exported Files")

                ex_btn.click(
                    fn=_do_export,
                    inputs=[ex_video, ex_query, ex_top_k, ex_threshold, ex_model, ex_dir],
                    outputs=ex_output,
                )

            # ------------------------------------------------------------------
            # Tab 4 — Video Info
            # ------------------------------------------------------------------
            with gr.TabItem("ℹ️ Video Info"):
                with gr.Row():
                    with gr.Column():
                        vi_video = gr.Video(label="Upload Video")
                        vi_btn = gr.Button("Get Info", variant="primary")
                    with gr.Column():
                        vi_output = gr.Markdown(label="Metadata")

                vi_btn.click(
                    fn=_do_video_info,
                    inputs=[vi_video],
                    outputs=vi_output,
                )

            # ------------------------------------------------------------------
            # Tab 5 — Help
            # ------------------------------------------------------------------
            with gr.TabItem("❓ Help"):
                gr.Markdown(
                    """
## How to use VideoSeek-CLI

### Search Tab
1. Upload a video file
2. Type a natural language query (e.g. *"Find the car crash"*) or pick one from **Example Queries**
3. Adjust **Top K** (how many results) and **Threshold** (how strict the matching is)
4. Pick a **Vision model** — budget models are fast and cheap, premium models are more accurate
5. Click **Search** — results and live stats appear on the right

### Batch Search Tab
Run multiple queries in a single indexing pass. Enter one query per line — saves time and API cost.

### Export Frames Tab
Search for moments and save the matched video frames as JPEG images to a local directory.

### Video Info Tab
Inspect video metadata: duration, FPS, resolution, total frames.

---

### Available Models

**Premium (highest quality)**
"""
                    + "\n".join(f"- `{m}`" for m in PREMIUM_MODELS)
                    + """

**Budget (fastest / most economical)**
"""
                    + "\n".join(f"- `{m}`" for m in BUDGET_MODELS)
                    + """

---

### CLI Usage
```bash
# Single search
videoseek-cli search -v video.mp4 -q "Find the car crash" --show-stats

# Batch search with report
videoseek-cli batch-search -v video.mp4 -q "sunset" -q "explosion" --report out.html

# Generate full report from a queries file
videoseek-cli report -v video.mp4 -Q queries.txt -o report.html

# List all available models
videoseek-cli list-models
```

---
*Built autonomously by [NEO](https://heyneo.so) — your autonomous AI Agent*
"""
                )

        gr.Markdown(
            "---\n"
            "Built autonomously by [NEO](https://heyneo.so) — your autonomous AI Agent  |  "
            "[GitHub](https://github.com/neo/VideoSeek-CLI)  |  "
            "[OpenRouter](https://openrouter.ai)"
        )

    return demo


def launch(share: bool = False, server_port: int | None = None) -> None:
    """Launch the Gradio web UI on the configured port."""
    port = server_port or get_gradio_port()
    ui = build_ui()
    ui.launch(server_port=port, share=share)
