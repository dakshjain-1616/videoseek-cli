"""Tests for the reporter module (markdown, HTML, JSON report generation)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from videoseek_cli.analyzer import SearchResult
from videoseek_cli.reporter import (
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
    write_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(ts_str: str, ts_sec: float, score: float, desc: str) -> SearchResult:
    return SearchResult(
        timestamp_str=ts_str,
        timestamp_seconds=ts_sec,
        frame_index=int(ts_sec),
        description=desc,
        similarity_score=score,
    )


def _sample_results_map() -> dict[str, list[SearchResult]]:
    return {
        "Find the car crash": [
            _make_result("01:23:45", 5025.0, 0.91, "A dramatic car collision on a highway."),
            _make_result("00:10:00", 600.0, 0.55, "A minor fender-bender in a parking lot."),
        ],
        "Show the sunset": [
            _make_result("00:45:30", 2730.0, 0.88, "A beautiful sunset over the ocean."),
        ],
        "Find a dragon": [],
    }


_VIDEO = "/demo/video.mp4"
_STATS = {
    "frames_indexed": 50,
    "index_time_seconds": 1.23,
    "query_time_seconds": 0.005,
    "total_time_seconds": 1.235,
    "total_tokens": 2000,
    "api_calls": 50,
}
_VIDEO_INFO = {
    "total_frames": 5000,
    "native_fps": 25.0,
    "width": 1920,
    "height": 1080,
    "duration_seconds": 200.0,
    "duration_str": "00:03:20",
}


# ---------------------------------------------------------------------------
# Markdown report tests
# ---------------------------------------------------------------------------

class TestMarkdownReport:
    def test_contains_video_path(self):
        md = generate_markdown_report(_VIDEO, _sample_results_map())
        assert _VIDEO in md

    def test_contains_all_queries(self):
        md = generate_markdown_report(_VIDEO, _sample_results_map())
        for q in _sample_results_map():
            assert q in md

    def test_contains_timestamps(self):
        md = generate_markdown_report(_VIDEO, _sample_results_map())
        assert "01:23:45" in md
        assert "00:45:30" in md

    def test_no_match_shown(self):
        md = generate_markdown_report(_VIDEO, _sample_results_map())
        assert "No matching moment found" in md

    def test_includes_stats_section(self):
        md = generate_markdown_report(_VIDEO, _sample_results_map(), stats=_STATS)
        assert "Analysis Stats" in md
        assert "50" in md  # frames_indexed

    def test_includes_video_info_section(self):
        md = generate_markdown_report(
            _VIDEO, _sample_results_map(), video_info=_VIDEO_INFO
        )
        assert "Video Information" in md
        assert "1920" in md

    def test_scores_present(self):
        md = generate_markdown_report(_VIDEO, _sample_results_map())
        assert "0.91" in md
        assert "0.88" in md

    def test_returns_string(self):
        md = generate_markdown_report(_VIDEO, _sample_results_map())
        assert isinstance(md, str)
        assert len(md) > 100


# ---------------------------------------------------------------------------
# HTML report tests
# ---------------------------------------------------------------------------

class TestHtmlReport:
    def test_is_valid_html_structure(self):
        html = generate_html_report(_VIDEO, _sample_results_map())
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_contains_video_path(self):
        html = generate_html_report(_VIDEO, _sample_results_map())
        assert "/demo/video.mp4" in html

    def test_contains_timestamps(self):
        html = generate_html_report(_VIDEO, _sample_results_map())
        assert "01:23:45" in html
        assert "00:45:30" in html

    def test_no_match_shown(self):
        html = generate_html_report(_VIDEO, _sample_results_map())
        assert "No matching moment found" in html

    def test_includes_stats_cards(self):
        html = generate_html_report(_VIDEO, _sample_results_map(), stats=_STATS)
        assert "stat-card" in html
        assert "50" in html

    def test_includes_video_info_table(self):
        html = generate_html_report(
            _VIDEO, _sample_results_map(), video_info=_VIDEO_INFO
        )
        assert "Video Information" in html
        assert "1920" in html

    def test_xss_escaping(self):
        malicious = {"<script>alert(1)</script>": []}
        html = generate_html_report(_VIDEO, malicious)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# JSON report tests
# ---------------------------------------------------------------------------

class TestJsonReport:
    def test_valid_json(self):
        js = generate_json_report(_VIDEO, _sample_results_map())
        data = json.loads(js)
        assert isinstance(data, dict)

    def test_contains_video_key(self):
        js = generate_json_report(_VIDEO, _sample_results_map())
        data = json.loads(js)
        assert data["video"] == _VIDEO

    def test_queries_key_present(self):
        js = generate_json_report(_VIDEO, _sample_results_map())
        data = json.loads(js)
        assert "queries" in data

    def test_all_queries_present(self):
        js = generate_json_report(_VIDEO, _sample_results_map())
        data = json.loads(js)
        for q in _sample_results_map():
            assert q in data["queries"]

    def test_results_structure(self):
        js = generate_json_report(_VIDEO, _sample_results_map())
        data = json.loads(js)
        hits = data["queries"]["Find the car crash"]
        assert isinstance(hits, list)
        assert len(hits) == 2
        first = hits[0]
        assert first["rank"] == 1
        assert first["timestamp"] == "01:23:45"
        assert first["score"] == 0.91

    def test_empty_query_returns_empty_list(self):
        js = generate_json_report(_VIDEO, _sample_results_map())
        data = json.loads(js)
        assert data["queries"]["Find a dragon"] == []

    def test_stats_included_when_provided(self):
        js = generate_json_report(_VIDEO, _sample_results_map(), stats=_STATS)
        data = json.loads(js)
        assert "stats" in data
        assert data["stats"]["frames_indexed"] == 50

    def test_video_info_included_when_provided(self):
        js = generate_json_report(_VIDEO, _sample_results_map(), video_info=_VIDEO_INFO)
        data = json.loads(js)
        assert "video_info" in data
        assert data["video_info"]["width"] == 1920


# ---------------------------------------------------------------------------
# write_report (file I/O) tests
# ---------------------------------------------------------------------------

class TestWriteReport:
    def test_writes_markdown_file(self, tmp_path):
        out = str(tmp_path / "report.md")
        path = write_report(_VIDEO, _sample_results_map(), output_path=out, fmt="markdown")
        assert Path(path).exists()
        assert Path(path).read_text().startswith("# VideoSeek-CLI")

    def test_writes_html_file(self, tmp_path):
        out = str(tmp_path / "report.html")
        path = write_report(_VIDEO, _sample_results_map(), output_path=out, fmt="html")
        assert Path(path).exists()
        content = Path(path).read_text()
        assert "<!DOCTYPE html>" in content

    def test_writes_json_file(self, tmp_path):
        out = str(tmp_path / "report.json")
        path = write_report(_VIDEO, _sample_results_map(), output_path=out, fmt="json")
        assert Path(path).exists()
        data = json.loads(Path(path).read_text())
        assert data["video"] == _VIDEO

    def test_md_alias(self, tmp_path):
        out = str(tmp_path / "report.md")
        path = write_report(_VIDEO, _sample_results_map(), output_path=out, fmt="md")
        assert Path(path).exists()

    def test_invalid_format_raises(self, tmp_path):
        out = str(tmp_path / "report.xyz")
        with pytest.raises(ValueError, match="Unsupported report format"):
            write_report(_VIDEO, _sample_results_map(), output_path=out, fmt="xyz")

    def test_creates_parent_dirs(self, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "report.md")
        write_report(_VIDEO, _sample_results_map(), output_path=out, fmt="markdown")
        assert Path(out).exists()
