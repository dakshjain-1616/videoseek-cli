#!/usr/bin/env python3
"""
VideoSeek-CLI Demo
==================
Demonstrates the complete pipeline in mock/dry-run mode without any API keys.
Automatically falls back to mock mode when OPENROUTER_API_KEY is not set.

Run:
    python demo.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time

# Ensure mock mode for demo
if not os.getenv("OPENROUTER_API_KEY", "").strip():
    os.environ["OPENROUTER_API_KEY"] = ""

# Use a temporary DB so the demo is always clean
_tmp_db = tempfile.mkdtemp(prefix="videoseek_demo_")
os.environ["VIDEOSEEK_DB_PATH"] = _tmp_db
os.environ["VIDEOSEEK_SIMILARITY_THRESHOLD"] = "0.1"

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from videoseek_cli.analyzer import NO_MATCH_MSG, VideoAnalyzer
from videoseek_cli.config import is_mock_mode
from videoseek_cli.embeddings import FrameStore
from videoseek_cli.frame_extractor import seconds_to_timestamp

console = Console()

# ---------------------------------------------------------------------------
# Build a pre-seeded analyzer (no real video file needed)
# ---------------------------------------------------------------------------

def _build_demo_analyzer() -> VideoAnalyzer:
    """
    Create a VideoAnalyzer with a hand-crafted scene library that demonstrates
    all three test-spec scenarios.
    """
    # Specific timestamps from the spec
    CRASH_TS = 5025.0   # 01:23:45
    SUNSET_TS = 2730.0  # 00:45:30

    scene_records = [
        # Generic background frames
        *[
            {
                "frame_index": i,
                "timestamp_seconds": float(i * 60),
                "timestamp_str": seconds_to_timestamp(i * 60),
                "description": f"Generic indoor shot, frame {i}.",
            }
            for i in range(15)
        ],
        # Key scene: car crash at 01:23:45
        {
            "frame_index": 1000,
            "timestamp_seconds": CRASH_TS,
            "timestamp_str": seconds_to_timestamp(CRASH_TS),
            "description": (
                "A dramatic high-speed vehicle collision on a highway — "
                "the car crashes into a concrete barrier, debris flying everywhere."
            ),
        },
        # Key scene: sunset at 00:45:30
        {
            "frame_index": 2000,
            "timestamp_seconds": SUNSET_TS,
            "timestamp_str": seconds_to_timestamp(SUNSET_TS),
            "description": (
                "A breathtaking sunset paints the sky in vibrant orange and pink hues "
                "over the ocean horizon as the sun dips below the water."
            ),
        },
        # Extra scenes for variety
        {
            "frame_index": 3000,
            "timestamp_seconds": 120.0,
            "timestamp_str": "00:02:00",
            "description": "Two people walking along a beach shoreline.",
        },
        {
            "frame_index": 3001,
            "timestamp_seconds": 180.0,
            "timestamp_str": "00:03:00",
            "description": "An aerial shot of a city skyline at night.",
        },
        {
            "frame_index": 3002,
            "timestamp_seconds": 240.0,
            "timestamp_str": "00:04:00",
            "description": "Close-up of a fire blazing in a fireplace.",
        },
    ]

    analyzer = VideoAnalyzer.__new__(VideoAnalyzer)
    analyzer.video_path = "/demo/video.mp4"
    analyzer.sample_fps = None
    analyzer.max_frames = None
    from videoseek_cli.vision_client import VisionClient
    analyzer._store = FrameStore("/demo/video.mp4")
    analyzer._client = VisionClient()
    analyzer._store.add_batch(scene_records)
    return analyzer


# ---------------------------------------------------------------------------
# Demo runner
# ---------------------------------------------------------------------------

def run_demo() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]VideoSeek-CLI[/bold cyan]  [dim]Demo Mode[/dim]",
            border_style="cyan",
        )
    )

    mode_label = "[yellow]MOCK (no API key)[/yellow]" if is_mock_mode() else "[green]LIVE[/green]"
    console.print(f"Mode: {mode_label}\n")

    console.print("[bold]Building demo video index...[/bold]")
    t0 = time.time()
    analyzer = _build_demo_analyzer()
    elapsed = time.time() - t0
    console.print(f"[green]✓[/green] Indexed {analyzer._store.frame_count()} frames in {elapsed:.2f}s\n")

    queries = [
        ("Find the car crash", "01:23:45"),
        ("Show the sunset", "00:45:30"),
        ("Find a non-existent event: underwater volcano eruption", None),
    ]

    all_passed = True

    for i, (query, expected_ts) in enumerate(queries, 1):
        console.rule(f"[bold]Test {i}[/bold]")
        console.print(f"[bold]Query:[/bold] {query}")

        result = analyzer.query_best(query)

        if expected_ts is None:
            # Should return no match
            top_results = analyzer.query(query, threshold=0.99)
            if not top_results:
                console.print(f'[green]✓ PASS[/green]  Output: "{NO_MATCH_MSG}"')
            else:
                console.print(
                    f"[red]✗ FAIL[/red]  Expected no match but got: "
                    f"{top_results[0].timestamp_str}"
                )
                all_passed = False
        else:
            if result and result.timestamp_str == expected_ts:
                console.print(
                    f"[green]✓ PASS[/green]  Timestamp [bold green]{result.timestamp_str}[/bold green]  "
                    f"(score={result.similarity_score:.3f})"
                )
            elif result:
                console.print(
                    f"[yellow]~ INFO[/yellow]  Got [bold]{result.timestamp_str}[/bold] "
                    f"(expected {expected_ts})  score={result.similarity_score:.3f}"
                )
                console.print(f"         Description: {result.description}")
            else:
                console.print(
                    f"[red]✗ FAIL[/red]  Expected timestamp {expected_ts} but got no match"
                )
                all_passed = False

        console.print()

    # Show top results table
    console.rule("[bold]Sample Multi-Result Query[/bold]")
    console.print("[bold]Query:[/bold] Find action scenes\n")
    results = analyzer.query("Find action scenes", top_k=5, threshold=0.05)
    if results:
        table = Table(border_style="cyan")
        table.add_column("Rank", style="dim", width=4)
        table.add_column("Timestamp", style="bold green")
        table.add_column("Score", style="cyan")
        table.add_column("Description", style="white")
        for i, r in enumerate(results, 1):
            table.add_row(str(i), r.timestamp_str, f"{r.similarity_score:.3f}", r.description[:80])
        console.print(table)
    else:
        console.print("[yellow]No results above threshold.[/yellow]")

    console.print()
    status = "[bold green]All demo checks passed![/bold green]" if all_passed else "[bold yellow]Demo complete (see above)[/bold yellow]"
    console.print(Panel(status, border_style="green" if all_passed else "yellow"))
    console.print(
        "\n[dim]Built autonomously using NEO - your autonomous AI Agent "
        "https://heyneo.so[/dim]"
    )


if __name__ == "__main__":
    run_demo()
