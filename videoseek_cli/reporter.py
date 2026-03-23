"""Report generation for VideoSeek-CLI analysis results."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import SearchResult


_NO_MATCH = "No matching moment found"


def generate_markdown_report(
    video_path: str,
    results_map: dict[str, list["SearchResult"]],
    stats: dict | None = None,
    video_info: dict | None = None,
) -> str:
    """
    Generate a markdown-formatted report from batch query results.

    Args:
        video_path: Path (or label) for the analysed video.
        results_map: Mapping of query_text → list[SearchResult].
        stats: Optional runtime stats dict from analyzer.get_stats().
        video_info: Optional video metadata dict from analyzer.get_video_info().

    Returns:
        A markdown string ready to be written to a .md file.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# VideoSeek-CLI Analysis Report",
        f"\n**Video:** `{video_path}`  ",
        f"**Generated:** {now}  ",
        f"**Queries:** {len(results_map)}",
    ]

    if video_info:
        lines.append("\n## Video Information")
        lines.append("| Property | Value |")
        lines.append("|---|---|")
        for k, v in video_info.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

    if stats:
        lines.append("\n## Analysis Stats")
        lines.append("| Metric | Value |")
        lines.append("|---|---|")
        for k, v in stats.items():
            lines.append(f"| {k.replace('_', ' ').title()} | {v} |")

    lines.append("\n## Query Results")

    for query, results in results_map.items():
        lines.append(f"\n### {query}")
        if not results:
            lines.append(f"> {_NO_MATCH}")
        else:
            lines.append(f"**Best match:** Timestamp `{results[0].timestamp_str}` "
                         f"(score={results[0].similarity_score:.3f})\n")
            lines.append("| # | Timestamp | Score | Description |")
            lines.append("|---|---|---|---|")
            for i, r in enumerate(results, 1):
                desc = r.description.replace("|", "\\|")
                lines.append(
                    f"| {i} | `{r.timestamp_str}` | {r.similarity_score:.3f} | {desc} |"
                )

    lines.append(
        "\n---\n*Built with [VideoSeek-CLI](https://github.com/neo/VideoSeek-CLI)*"
    )
    return "\n".join(lines) + "\n"

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>VideoSeek-CLI Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         max-width: 960px; margin: 2rem auto; padding: 0 1.5rem; color: #222; }}
  h1 {{ color: #0a84ff; border-bottom: 2px solid #0a84ff; padding-bottom: .4rem; }}
  h2 {{ color: #555; margin-top: 2rem; }}
  h3 {{ color: #333; border-left: 4px solid #0a84ff; padding-left: .6rem; }}
  .meta {{ color: #666; font-size: .9rem; }}
  table {{ border-collapse: collapse; width: 100%; margin-top: .8rem; }}
  th {{ background: #0a84ff; color: #fff; padding: .5rem .8rem; text-align: left; }}
  td {{ padding: .45rem .8rem; border-bottom: 1px solid #eee; }}
  tr:hover td {{ background: #f0f6ff; }}
  .score-bar {{ display: inline-block; height: .6rem; background: #0a84ff;
               border-radius: 3px; vertical-align: middle; margin-left: .4rem; }}
  .no-match {{ color: #999; font-style: italic; }}
  .best {{ background: #e6f4ea; font-weight: bold; }}
  footer {{ margin-top: 3rem; color: #aaa; font-size: .8rem; text-align: center; }}
  .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px,1fr));
                 gap: .8rem; margin: 1rem 0; }}
  .stat-card {{ background: #f7f9fc; border: 1px solid #e0e7ef; border-radius: 8px;
                padding: .8rem 1rem; }}
  .stat-card .label {{ font-size: .78rem; color: #888; text-transform: uppercase; }}
  .stat-card .value {{ font-size: 1.4rem; font-weight: bold; color: #0a84ff; }}
</style>
</head>
<body>
<h1>🎬 VideoSeek-CLI Analysis Report</h1>
<p class="meta"><strong>Video:</strong> {video_path}<br>
<strong>Generated:</strong> {now}<br>
<strong>Queries:</strong> {num_queries}</p>

{video_info_section}
{stats_section}

<h2>Query Results</h2>
{results_sections}

<footer>Built with <a href="https://github.com/neo/VideoSeek-CLI">VideoSeek-CLI</a></footer>
</body>
</html>
"""


def generate_html_report(
    video_path: str,
    results_map: dict[str, list["SearchResult"]],
    stats: dict | None = None,
    video_info: dict | None = None,
) -> str:
    """Generate a self-contained HTML report from batch query results."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Video info section
    vi_html = ""
    if video_info:
        rows = "".join(
            f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v}</td></tr>"
            for k, v in video_info.items()
        )
        vi_html = (
            "<h2>Video Information</h2>"
            "<table><tr><th>Property</th><th>Value</th></tr>"
            f"{rows}</table>"
        )

    # Stats section
    st_html = ""
    if stats:
        cards = ""
        for k, v in stats.items():
            label = k.replace("_", " ").title()
            cards += (
                f'<div class="stat-card">'
                f'<div class="label">{label}</div>'
                f'<div class="value">{v}</div></div>'
            )
        st_html = f"<h2>Analysis Stats</h2><div class='stats-grid'>{cards}</div>"

    # Results sections
    res_html = ""
    for query, results in results_map.items():
        res_html += f"<h3>{_esc(query)}</h3>"
        if not results:
            res_html += f'<p class="no-match">{_NO_MATCH}</p>'
        else:
            rows = ""
            for i, r in enumerate(results, 1):
                bar_w = int(r.similarity_score * 80)
                row_cls = ' class="best"' if i == 1 else ""
                rows += (
                    f"<tr{row_cls}><td>{i}</td>"
                    f"<td><code>{r.timestamp_str}</code></td>"
                    f"<td>{r.similarity_score:.3f}"
                    f'<span class="score-bar" style="width:{bar_w}px"></span></td>'
                    f"<td>{_esc(r.description)}</td></tr>"
                )
            res_html += (
                "<table>"
                "<tr><th>#</th><th>Timestamp</th><th>Score</th><th>Description</th></tr>"
                f"{rows}</table>"
            )

    return _HTML_TEMPLATE.format(
        video_path=_esc(video_path),
        now=now,
        num_queries=len(results_map),
        video_info_section=vi_html,
        stats_section=st_html,
        results_sections=res_html,
    )


def generate_json_report(
    video_path: str,
    results_map: dict[str, list["SearchResult"]],
    stats: dict | None = None,
    video_info: dict | None = None,
) -> str:
    """Generate a JSON report (pretty-printed string) from batch query results."""
    payload: dict = {
        "video": video_path,
        "generated": datetime.now().isoformat(),
        "queries": {},
    }
    if video_info:
        payload["video_info"] = video_info
    if stats:
        payload["stats"] = stats

    for query, results in results_map.items():
        payload["queries"][query] = [
            {
                "rank": i,
                "timestamp": r.timestamp_str,
                "timestamp_seconds": r.timestamp_seconds,
                "frame_index": r.frame_index,
                "description": r.description,
                "score": round(r.similarity_score, 4),
            }
            for i, r in enumerate(results, 1)
        ] if results else []

    return json.dumps(payload, indent=2, ensure_ascii=False)


def write_report(
    video_path: str,
    results_map: dict[str, list["SearchResult"]],
    output_path: str,
    fmt: str = "markdown",
    stats: dict | None = None,
    video_info: dict | None = None,
) -> str:
    """
    Write a report to *output_path* in the given format ('markdown', 'html', 'json').
    Returns the resolved output path.
    """
    generators = {
        "markdown": generate_markdown_report,
        "md": generate_markdown_report,
        "html": generate_html_report,
        "json": generate_json_report,
    }
    gen = generators.get(fmt.lower())
    if gen is None:
        raise ValueError(f"Unsupported report format: {fmt!r}. Choose markdown/html/json.")

    content = gen(video_path, results_map, stats=stats, video_info=video_info)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    return str(out.resolve())


def _esc(text: str) -> str:
    """Minimal HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
