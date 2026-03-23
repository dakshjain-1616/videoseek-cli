"""VideoSeek-CLI: Extract key moments from videos using natural language queries."""

__version__ = "1.0.0"
__author__ = "NEO"

from .analyzer import NO_MATCH_MSG, AnalysisStats, SearchResult, VideoAnalyzer
from .config import is_mock_mode
from .frame_extractor import seconds_to_timestamp
from .reporter import generate_json_report, generate_markdown_report, write_report

# Heavy optional modules (server, gradio_ui) are not imported here to avoid
# pulling in fastapi / gradio as mandatory dependencies.

__all__ = [
    "VideoAnalyzer",
    "SearchResult",
    "AnalysisStats",
    "NO_MATCH_MSG",
    "is_mock_mode",
    "seconds_to_timestamp",
    "generate_markdown_report",
    "generate_json_report",
    "write_report",
]
