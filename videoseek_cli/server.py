"""FastAPI REST server for VideoSeek-CLI."""

from __future__ import annotations

import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .analyzer import NO_MATCH_MSG, VideoAnalyzer
from .config import get_host, get_port, is_mock_mode

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VideoSeek-CLI API",
    description="Long-horizon video moment extraction via natural language queries.",
    version="1.0.0",
)


class SearchRequest(BaseModel):
    video: str
    query: str
    top_k: int = 3
    threshold: float | None = None
    force_reindex: bool = False


class SearchResponse(BaseModel):
    query: str
    video: str
    results: list[dict]
    message: str | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mock_mode": is_mock_mode()}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not is_mock_mode() and not Path(req.video).is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {req.video}")

    analyzer = VideoAnalyzer(
        video_path=req.video,
        force_reindex=req.force_reindex,
    )
    analyzer.index()
    results = analyzer.query(req.query, top_k=req.top_k, threshold=req.threshold)

    payload_results = [
        {
            "timestamp": r.timestamp_str,
            "timestamp_seconds": r.timestamp_seconds,
            "frame_index": r.frame_index,
            "description": r.description,
            "score": round(r.similarity_score, 4),
        }
        for r in results
    ]

    return SearchResponse(
        query=req.query,
        video=req.video,
        results=payload_results,
        message=None if results else NO_MATCH_MSG,
    )


@app.get("/info")
def video_info(video: str) -> dict:
    if not is_mock_mode() and not Path(video).is_file():
        raise HTTPException(status_code=404, detail=f"Video not found: {video}")
    analyzer = VideoAnalyzer(video_path=video)
    return analyzer.get_video_info()


def run_server(host: str | None = None, port: int | None = None) -> None:
    h = host or get_host()
    p = port or get_port()
    logger.info("Starting VideoSeek-CLI API server at http://%s:%d", h, p)
    uvicorn.run(app, host=h, port=p, log_level="info")
