"""VideoSeek-CLI: command-line interface entry point."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .analyzer import NO_MATCH_MSG, VideoAnalyzer
from .config import is_mock_mode

console = Console()


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def _print_banner() -> None:
    console.print(
        Panel.fit(
            "[bold cyan]VideoSeek-CLI[/bold cyan]  "
            "[dim]Long-horizon video moment extraction[/dim]",
            border_style="cyan",
        )
    )
    if is_mock_mode():
        console.print(
            "[yellow]⚠  Running in mock/demo mode "
            "(set OPENROUTER_API_KEY for real analysis)[/yellow]\n"
        )


def _print_stats(stats: dict) -> None:
    """Print a compact stats panel after a search."""
    parts = []
    if "frames_indexed" in stats:
        parts.append(f"frames={stats['frames_indexed']}")
    if "index_time_seconds" in stats:
        parts.append(f"index={stats['index_time_seconds']:.2f}s")
    if "query_time_seconds" in stats:
        parts.append(f"query={stats['query_time_seconds']:.3f}s")
    if stats.get("total_tokens"):
        parts.append(f"tokens={stats['total_tokens']}")
    if stats.get("api_calls"):
        parts.append(f"api_calls={stats['api_calls']}")
    if parts:
        console.print(f"\n[dim]Stats: {' | '.join(parts)}[/dim]")


@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    """VideoSeek-CLI — find moments in videos using natural language."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.option("--video", "-v", required=True, help="Path to the video file.")
@click.option("--query", "-q", required=True, help="Natural language query.")
@click.option(
    "--top-k",
    "-k",
    default=3,
    show_default=True,
    help="Number of results to return.",
)
@click.option(
    "--threshold",
    "-t",
    default=None,
    type=float,
    help="Similarity threshold override (0-1).",
)
@click.option(
    "--fps",
    default=None,
    type=float,
    help="Frames-per-second to sample (overrides VIDEOSEEK_FPS).",
)
@click.option(
    "--model",
    "-m",
    default=None,
    help="OpenRouter model ID to use (overrides VIDEOSEEK_MODEL).",
)
@click.option(
    "--force-reindex",
    is_flag=True,
    default=False,
    help="Re-index the video even if a cached index exists.",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--export-frames",
    is_flag=True,
    default=False,
    help="Save matched frames as images to VIDEOSEEK_EXPORT_DIR.",
)
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
@click.option("--show-stats", is_flag=True, default=False, help="Print runtime stats after search.")
def search(video, query, top_k, threshold, fps, model, force_reindex, output, export_frames, debug, show_stats):
    """Search a video for moments matching a natural language query."""
    _setup_logging(debug)
    if debug:
        os.environ["VIDEOSEEK_DEBUG"] = "true"

    if output == "text":
        _print_banner()

    # Validate video path (skip for mock mode where no real file is needed)
    if not is_mock_mode() and not Path(video).is_file():
        console.print(f"[red]Error: video file not found: {video}[/red]")
        sys.exit(1)

    analyzer = VideoAnalyzer(
        video_path=video,
        sample_fps=fps,
        force_reindex=force_reindex,
        model=model,
    )

    # Index phase
    if output == "text":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Indexing frames...", total=None)

            def _cb(done, total):
                progress.update(task, completed=done, total=total)

            count = analyzer.index(progress_callback=_cb)
        console.print(f"[green]✓[/green] Indexed [bold]{count}[/bold] frames.")
    else:
        analyzer.index()

    # Query phase
    results = analyzer.query(query, top_k=top_k, threshold=threshold)

    # Export frames if requested
    if export_frames and results:
        exported = analyzer.export_frames(results)
        if output == "text" and exported:
            console.print(
                f"[green]✓[/green] Exported [bold]{len(exported)}[/bold] frames to [cyan]{exported[0]}[/cyan] ..."
            )

    if output == "json":
        stats = analyzer.get_stats() if show_stats else {}
        payload = {
            "query": query,
            "video": video,
            "results": (
                [
                    {
                        "timestamp": r.timestamp_str,
                        "timestamp_seconds": r.timestamp_seconds,
                        "frame_index": r.frame_index,
                        "description": r.description,
                        "score": round(r.similarity_score, 4),
                    }
                    for r in results
                ]
                if results
                else []
            ),
            "message": None if results else NO_MATCH_MSG,
        }
        if stats:
            payload["stats"] = stats
        click.echo(json.dumps(payload, indent=2))
        return

    # Rich text output
    if not results:
        console.print(
            Panel(
                f"[yellow]{NO_MATCH_MSG}[/yellow]\n\n"
                "Try a different query or lower --threshold.",
                title="[bold]Query Result[/bold]",
                border_style="yellow",
            )
        )
        if show_stats:
            _print_stats(analyzer.get_stats())
        return

    table = Table(title=f"Results for: [italic]{query}[/italic]", border_style="cyan")
    table.add_column("Rank", style="dim", width=4)
    table.add_column("Timestamp", style="bold green")
    table.add_column("Score", style="cyan")
    table.add_column("Frame #", style="dim")
    table.add_column("Description", style="white")

    for i, r in enumerate(results, 1):
        table.add_row(
            str(i),
            r.timestamp_str,
            f"{r.similarity_score:.3f}",
            str(r.frame_index),
            r.description,
        )

    console.print(table)
    console.print(
        f"\n[bold]Best match:[/bold] "
        f"[bold green]Timestamp {results[0].timestamp_str}[/bold green]  "
        f"[dim](score={results[0].similarity_score:.3f})[/dim]"
    )

    if show_stats:
        _print_stats(analyzer.get_stats())


@main.command("batch-search")
@click.option("--video", "-v", required=True, help="Path to the video file.")
@click.option(
    "--queries-file",
    "-Q",
    default=None,
    help="Path to a plain-text file with one query per line.",
)
@click.option(
    "--query",
    "-q",
    multiple=True,
    help="Query string (repeatable, e.g. -q 'car crash' -q 'sunset').",
)
@click.option("--top-k", "-k", default=3, show_default=True)
@click.option("--threshold", "-t", default=None, type=float)
@click.option(
    "--model", "-m", default=None, help="OpenRouter model ID (overrides VIDEOSEEK_MODEL)."
)
@click.option(
    "--report",
    "-r",
    default=None,
    help="Write a report to this path (extension determines format: .md/.html/.json).",
)
@click.option(
    "--output",
    "-o",
    type=click.Choice(["text", "json"]),
    default="text",
    show_default=True,
)
@click.option("--force-reindex", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
def batch_search(video, queries_file, query, top_k, threshold, model, report, output, force_reindex, debug):
    """Run multiple queries against a video in a single indexing pass."""
    _setup_logging(debug)
    if debug:
        os.environ["VIDEOSEEK_DEBUG"] = "true"

    if output == "text":
        _print_banner()

    # Collect queries
    queries: list[str] = list(query)
    if queries_file:
        qf = Path(queries_file)
        if not qf.is_file():
            console.print(f"[red]Error: queries file not found: {queries_file}[/red]")
            sys.exit(1)
        for line in qf.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                queries.append(line)

    if not queries:
        console.print("[red]Error: provide at least one query via --query or --queries-file[/red]")
        sys.exit(1)

    if not is_mock_mode() and not Path(video).is_file():
        console.print(f"[red]Error: video file not found: {video}[/red]")
        sys.exit(1)

    analyzer = VideoAnalyzer(
        video_path=video,
        force_reindex=force_reindex,
        model=model,
    )

    # Index phase
    if output == "text":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TimeElapsedColumn(),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Indexing frames...", total=None)

            def _cb(done, total):
                progress.update(task, completed=done, total=total)

            count = analyzer.index(progress_callback=_cb)
        console.print(f"[green]✓[/green] Indexed [bold]{count}[/bold] frames.")
    else:
        analyzer.index()

    # Batch query
    results_map = analyzer.batch_query(queries, top_k=top_k, threshold=threshold)

    if output == "json":
        payload = {
            "video": video,
            "results": {
                q: [
                    {
                        "timestamp": r.timestamp_str,
                        "timestamp_seconds": r.timestamp_seconds,
                        "frame_index": r.frame_index,
                        "description": r.description,
                        "score": round(r.similarity_score, 4),
                    }
                    for r in rs
                ]
                for q, rs in results_map.items()
            },
            "stats": analyzer.get_stats(),
        }
        click.echo(json.dumps(payload, indent=2))
    else:
        for q, rs in results_map.items():
            console.rule(f"[bold]{q}[/bold]")
            if not rs:
                console.print(f"[yellow]{NO_MATCH_MSG}[/yellow]")
                continue
            table = Table(border_style="cyan", show_header=True)
            table.add_column("Rank", style="dim", width=4)
            table.add_column("Timestamp", style="bold green")
            table.add_column("Score", style="cyan")
            table.add_column("Description", style="white")
            for i, r in enumerate(rs, 1):
                table.add_row(str(i), r.timestamp_str, f"{r.similarity_score:.3f}", r.description)
            console.print(table)
        _print_stats(analyzer.get_stats())

    # Optional report export
    if report:
        from .reporter import write_report

        ext = Path(report).suffix.lstrip(".") or "markdown"
        fmt_map = {"md": "markdown", "html": "html", "json": "json"}
        fmt = fmt_map.get(ext, "markdown")
        video_info = analyzer.get_video_info()
        stats = analyzer.get_stats()
        path = write_report(
            video_path=video,
            results_map=results_map,
            output_path=report,
            fmt=fmt,
            stats=stats,
            video_info=video_info,
        )
        if output == "text":
            console.print(f"\n[green]✓[/green] Report written to [bold cyan]{path}[/bold cyan]")


@main.command()
@click.option("--video", "-v", required=True, help="Path to the video file.")
@click.option("--debug", is_flag=True, default=False, help="Enable debug logging.")
def info(video, debug):
    """Display metadata for a video file."""
    _setup_logging(debug)
    _print_banner()

    if not is_mock_mode() and not Path(video).is_file():
        console.print(f"[red]Error: video file not found: {video}[/red]")
        sys.exit(1)

    analyzer = VideoAnalyzer(video_path=video)
    meta = analyzer.get_video_info()

    table = Table(title="Video Info", border_style="cyan")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="green")
    for k, v in meta.items():
        table.add_row(k.replace("_", " ").title(), str(v))
    console.print(table)


@main.command()
@click.option("--video", "-v", required=True, help="Path to the video file.")
@click.option(
    "--queries-file",
    "-Q",
    required=True,
    help="Text file with one query per line.",
)
@click.option(
    "--output", "-o", required=True, help="Output path (.md, .html, or .json)."
)
@click.option("--top-k", "-k", default=3, show_default=True)
@click.option("--threshold", "-t", default=None, type=float)
@click.option("--model", "-m", default=None, help="OpenRouter model ID.")
@click.option("--force-reindex", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
def report(video, queries_file, output, top_k, threshold, model, force_reindex, debug):
    """Index a video, run queries from a file, and write a report."""
    _setup_logging(debug)
    _print_banner()

    qf = Path(queries_file)
    if not qf.is_file():
        console.print(f"[red]Error: queries file not found: {queries_file}[/red]")
        sys.exit(1)

    queries = [
        l.strip()
        for l in qf.read_text().splitlines()
        if l.strip() and not l.strip().startswith("#")
    ]
    if not queries:
        console.print("[red]Error: queries file is empty[/red]")
        sys.exit(1)

    if not is_mock_mode() and not Path(video).is_file():
        console.print(f"[red]Error: video file not found: {video}[/red]")
        sys.exit(1)

    analyzer = VideoAnalyzer(
        video_path=video,
        force_reindex=force_reindex,
        model=model,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Indexing frames...", total=None)

        def _cb(done, total):
            progress.update(task, completed=done, total=total)

        count = analyzer.index(progress_callback=_cb)
    console.print(f"[green]✓[/green] Indexed [bold]{count}[/bold] frames.")

    results_map = analyzer.batch_query(queries, top_k=top_k, threshold=threshold)

    from .reporter import write_report

    ext = Path(output).suffix.lstrip(".") or "markdown"
    fmt_map = {"md": "markdown", "html": "html", "json": "json"}
    fmt = fmt_map.get(ext, "markdown")

    path = write_report(
        video_path=video,
        results_map=results_map,
        output_path=output,
        fmt=fmt,
        stats=analyzer.get_stats(),
        video_info=analyzer.get_video_info(),
    )
    console.print(f"\n[green]✓[/green] Report written to [bold cyan]{path}[/bold cyan]")
    _print_stats(analyzer.get_stats())


@main.command()
@click.option("--host", default=None, help="Host to bind (overrides VIDEOSEEK_HOST).")
@click.option("--port", default=None, type=int, help="Port (overrides VIDEOSEEK_PORT).")
@click.option("--debug", is_flag=True, default=False)
def serve(host, port, debug):
    """Launch the FastAPI REST server."""
    _setup_logging(debug)
    from .server import run_server

    run_server(host=host, port=port)


@main.command()
@click.option("--port", default=None, type=int, help="Port (overrides VIDEOSEEK_GRADIO_PORT).")
@click.option("--share", is_flag=True, default=False, help="Create a public Gradio share link.")
@click.option("--debug", is_flag=True, default=False)
def ui(port, share, debug):
    """Launch the Gradio web UI."""
    _setup_logging(debug)
    from .gradio_ui import launch

    launch(share=share, server_port=port)


@main.command("list-models")
def list_models():
    """List available OpenRouter model IDs."""
    from .config import BUDGET_MODELS, PREMIUM_MODELS

    table = Table(title="Available Models", border_style="cyan")
    table.add_column("Tier", style="bold")
    table.add_column("Model ID", style="green")
    for m in PREMIUM_MODELS:
        table.add_row("Premium", m)
    for m in BUDGET_MODELS:
        table.add_row("Budget", m)
    console.print(table)
    console.print(
        "\n[dim]Set VIDEOSEEK_MODEL=<model-id> or use --model <model-id>[/dim]"
    )


# Allow `python -m videoseek_cli` invocation
if __name__ == "__main__":
    main()
