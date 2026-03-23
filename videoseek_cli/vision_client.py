"""OpenRouter vision API client for describing video frames."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

from .config import (
    get_max_tokens,
    get_model,
    get_openrouter_api_key,
    get_openrouter_base_url,
    get_retry_attempts,
    get_retry_delay,
    get_vision_detail,
    get_vision_timeout,
    is_debug,
    is_mock_mode,
)

logger = logging.getLogger(__name__)

# Prompt template for per-frame description
_FRAME_PROMPT = (
    "Describe this video frame in one concise sentence. "
    "Focus on the main action, subjects, setting, and any notable events. "
    "Be specific about what is visually present."
)


@dataclass
class TokenStats:
    """Cumulative token usage tracked across all API calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    api_calls: int = 0
    retries: int = 0

    def add(self, prompt: int, completion: int) -> None:
        """Accumulate token counts from a single API call."""
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.api_calls += 1

    def as_dict(self) -> dict:
        """Return token stats as a plain dict."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "retries": self.retries,
        }


class VisionClient:
    """Thin wrapper around OpenRouter's chat completions endpoint."""

    def __init__(self, model: str | None = None):
        self._base_url = get_openrouter_base_url()
        self._api_key = get_openrouter_api_key()
        self._model = model or get_model()
        self._max_tokens = get_max_tokens()
        self._detail = get_vision_detail()
        self._retry_attempts = get_retry_attempts()
        self._retry_delay = get_retry_delay()
        self._timeout = get_vision_timeout()
        self.token_stats = TokenStats()

    def describe_frame(self, image_b64: str, timestamp_str: str) -> str:
        """
        Send a frame to the vision model and return a short description.
        Falls back to a synthetic description in mock mode.
        Retries up to VIDEOSEEK_RETRY_ATTEMPTS times on transient errors.
        """
        if is_mock_mode():
            return _mock_description(timestamp_str)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/neo/VideoSeek-CLI",
            "X-Title": "VideoSeek-CLI",
        }
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}",
                                "detail": self._detail,
                            },
                        },
                        {"type": "text", "text": _FRAME_PROMPT},
                    ],
                }
            ],
            "max_tokens": self._max_tokens,
        }

        last_exc: Exception | None = None
        for attempt in range(1, self._retry_attempts + 1):
            try:
                with httpx.Client(timeout=self._timeout) as client:
                    resp = client.post(
                        f"{self._base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    description = data["choices"][0]["message"]["content"].strip()

                    # Track token usage when available
                    usage = data.get("usage", {})
                    self.token_stats.add(
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    )

                    if is_debug():
                        logger.debug("[%s] %s", timestamp_str, description)
                    return description

            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                last_exc = exc
                if attempt < self._retry_attempts:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Vision API transient error at %s (attempt %d/%d), retrying in %.1fs: %s",
                        timestamp_str,
                        attempt,
                        self._retry_attempts,
                        delay,
                        exc,
                    )
                    self.token_stats.retries += 1
                    time.sleep(delay)

            except httpx.HTTPStatusError as exc:
                # Non-retriable HTTP errors (4xx client errors)
                if exc.response.status_code < 500:
                    logger.warning(
                        "Vision API client error at %s: %s", timestamp_str, exc
                    )
                    return f"Frame at {timestamp_str}: visual content unavailable"
                last_exc = exc
                if attempt < self._retry_attempts:
                    delay = self._retry_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Vision API server error at %s (attempt %d/%d), retrying in %.1fs: %s",
                        timestamp_str,
                        attempt,
                        self._retry_attempts,
                        delay,
                        exc,
                    )
                    self.token_stats.retries += 1
                    time.sleep(delay)

            except Exception as exc:
                logger.warning("Vision API error at %s: %s", timestamp_str, exc)
                return f"Frame at {timestamp_str}: visual content unavailable"

        logger.warning(
            "Vision API failed after %d attempts at %s: %s",
            self._retry_attempts,
            timestamp_str,
            last_exc,
        )
        return f"Frame at {timestamp_str}: visual content unavailable"

    def batch_describe(
        self,
        frames: list[dict],
        progress_callback=None,
    ) -> list[dict]:
        """
        Describe a list of frame dicts with keys: {index, timestamp_seconds, timestamp_str, image_b64}.
        Returns list of {frame_index, timestamp_seconds, timestamp_str, description}.
        """
        results = []
        for i, frame in enumerate(frames):
            desc = self.describe_frame(frame["image_b64"], frame["timestamp_str"])
            results.append(
                {
                    "frame_index": frame["index"],
                    "timestamp_seconds": frame["timestamp_seconds"],
                    "timestamp_str": frame["timestamp_str"],
                    "description": desc,
                }
            )
            if progress_callback:
                progress_callback(i + 1, len(frames))
        return results

    def get_token_stats(self) -> dict:
        """Return cumulative token usage statistics."""
        return self.token_stats.as_dict()


# ---------------------------------------------------------------------------
# Mock / demo helpers
# ---------------------------------------------------------------------------

_MOCK_SCENE_LIBRARY = [
    # (keyword_triggers, description_template)
    (
        ["crash", "accident", "collision", "impact"],
        "A dramatic vehicle collision occurs on a busy road with debris flying through the air.",
    ),
    (
        ["sunset", "sunrise", "golden hour", "dusk", "dawn"],
        "A breathtaking sunset paints the sky in vibrant orange and pink hues over the horizon.",
    ),
    (
        ["car", "vehicle", "automobile", "driving"],
        "A car drives along a suburban street lined with trees.",
    ),
    (
        ["person", "people", "man", "woman", "human"],
        "Two people are seen walking and talking in a park.",
    ),
    (
        ["fire", "flame", "burning", "explosion"],
        "Bright orange flames engulf a structure as smoke billows into the sky.",
    ),
    (
        ["water", "ocean", "sea", "river", "lake"],
        "Clear blue water glistens under sunlight near a rocky shoreline.",
    ),
    (
        ["city", "urban", "street", "traffic"],
        "A busy urban intersection with heavy traffic and pedestrians crossing.",
    ),
    (
        ["sport", "game", "play", "athlete", "run"],
        "An athlete sprints across a field during an intense sports match.",
    ),
    (
        ["cook", "food", "kitchen", "meal", "eat"],
        "A chef prepares a meal in a professional kitchen with steaming pots.",
    ),
    (
        ["nature", "forest", "tree", "animal", "wildlife"],
        "A deer stands at the edge of a dense forest clearing.",
    ),
]

_DEFAULT_MOCK_DESCRIPTIONS = [
    "An exterior shot of a building with clear skies overhead.",
    "A wide landscape view showing rolling hills in the background.",
    "Indoor scene with people seated around a conference table.",
    "A close-up of hands typing on a keyboard.",
    "An overhead aerial view of a forest road.",
    "Static shot of an empty corridor lit by fluorescent lights.",
    "Two characters engaged in conversation in a living room.",
    "A panning shot across a crowded marketplace.",
    "Night-time shot of city lights reflected on wet pavement.",
    "A sports player runs across a green field.",
]


def _mock_description(timestamp_str: str) -> str:
    """Generate a deterministic pseudo-realistic frame description for demos."""
    # Use timestamp to pick a description so results are stable across runs
    h, m, s = (int(x) for x in timestamp_str.split(":"))
    total_seconds = h * 3600 + m * 60 + s
    idx = total_seconds % len(_DEFAULT_MOCK_DESCRIPTIONS)
    return _DEFAULT_MOCK_DESCRIPTIONS[idx]


def mock_descriptions_for_query(query: str, total_frames: int) -> list[dict]:
    """
    Generate a set of mock frame descriptions seeded by *query*, ensuring at
    least one frame matches the query concept so demos return useful results.
    """
    import random

    rng = random.Random(hash(query) % (2**31))
    query_lower = query.lower()

    # Find a matching scene template
    matching_desc = None
    for triggers, template in _MOCK_SCENE_LIBRARY:
        if any(kw in query_lower for kw in triggers):
            matching_desc = template
            break

    records = []
    # Place a matching frame roughly 1/3 through the video
    match_frame = max(0, total_frames // 3)

    for i in range(total_frames):
        ts_s = float(i)
        h = int(ts_s) // 3600
        m = (int(ts_s) % 3600) // 60
        s = int(ts_s) % 60
        ts_str = f"{h:02d}:{m:02d}:{s:02d}"

        if i == match_frame and matching_desc:
            desc = matching_desc
        else:
            desc = _DEFAULT_MOCK_DESCRIPTIONS[
                rng.randint(0, len(_DEFAULT_MOCK_DESCRIPTIONS) - 1)
            ]

        records.append(
            {
                "frame_index": i,
                "timestamp_seconds": ts_s,
                "timestamp_str": ts_str,
                "description": desc,
            }
        )
    return records
