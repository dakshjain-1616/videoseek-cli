"""Configuration management for VideoSeek-CLI."""

import os
from dotenv import load_dotenv

load_dotenv()

# Latest OpenRouter model IDs (updated 2026-03)
PREMIUM_MODELS = [
    "x-ai/grok-4.20-multi-agent-beta",
    "x-ai/grok-4.20-beta",
    "openai/gpt-5.4-pro",
    "openai/gpt-5.4",
    "google/gemini-3.1-pro-preview-customtools",
    "google/gemini-3.1-pro-preview",
]

BUDGET_MODELS = [
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash-lite-preview-09-2025",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "x-ai/grok-4-fast",
    "x-ai/grok-4.1-fast",
    "xiaomi/mimo-v2-pro",
]

ALL_MODELS = PREMIUM_MODELS + BUDGET_MODELS


def get_openrouter_api_key() -> str:
    """Return the OpenRouter API key from the environment."""
    return os.getenv("OPENROUTER_API_KEY", "")


def get_openrouter_base_url() -> str:
    """Return the OpenRouter API base URL."""
    return os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")


def get_model() -> str:
    """Return the vision model ID to use for frame analysis."""
    return os.getenv("VIDEOSEEK_MODEL", "google/gemini-2.5-flash")


def get_fps() -> float:
    """Return the number of frames per second to sample from the video."""
    return float(os.getenv("VIDEOSEEK_FPS", "1"))


def get_max_frames() -> int:
    """Return the maximum number of frames to analyze per query."""
    return int(os.getenv("VIDEOSEEK_MAX_FRAMES", "300"))


def get_similarity_threshold() -> float:
    """Return the minimum similarity score (0–1) required to include a result."""
    return float(os.getenv("VIDEOSEEK_SIMILARITY_THRESHOLD", "0.35"))


def get_db_path() -> str:
    """Return the directory path for the ChromaDB frame-embedding cache."""
    return os.getenv("VIDEOSEEK_DB_PATH", ".videoseek_db")


def get_host() -> str:
    """Return the host address for the FastAPI server."""
    return os.getenv("VIDEOSEEK_HOST", "0.0.0.0")


def get_port() -> int:
    """Return the port for the FastAPI server."""
    return int(os.getenv("VIDEOSEEK_PORT", "8000"))


def get_embedding_model() -> str:
    """Return the embedding model hint used for ChromaDB indexing."""
    return os.getenv("VIDEOSEEK_EMBEDDING_MODEL", "all-MiniLM-L6-v2")


def is_debug() -> bool:
    return os.getenv("VIDEOSEEK_DEBUG", "false").lower() in ("true", "1", "yes")


def is_mock_mode() -> bool:
    """Return True when no API key is configured — switches to demo/mock mode."""
    return not get_openrouter_api_key().strip()


def get_retry_attempts() -> int:
    """Number of times to retry failed API calls before giving up."""
    return int(os.getenv("VIDEOSEEK_RETRY_ATTEMPTS", "3"))


def get_retry_delay() -> float:
    """Base delay in seconds between retries (doubles each attempt)."""
    return float(os.getenv("VIDEOSEEK_RETRY_DELAY", "1.0"))


def get_max_tokens() -> int:
    """Maximum tokens the vision model may generate per frame description."""
    return int(os.getenv("VIDEOSEEK_MAX_TOKENS", "120"))


def get_vision_detail() -> str:
    """Vision API detail level: 'low', 'high', or 'auto'."""
    return os.getenv("VIDEOSEEK_VISION_DETAIL", "low")


def get_export_dir() -> str:
    """Directory where exported frame images are saved."""
    return os.getenv("VIDEOSEEK_EXPORT_DIR", "videoseek_exports")


def get_gradio_port() -> int:
    """Return the port for the Gradio web UI."""
    return int(os.getenv("VIDEOSEEK_GRADIO_PORT", "7860"))


def get_vision_timeout() -> int:
    """HTTP timeout in seconds for vision API calls."""
    return int(os.getenv("VIDEOSEEK_VISION_TIMEOUT", "30"))


def get_frame_size() -> int:
    """Maximum width/height (pixels) for frame thumbnails sent to the vision API."""
    return int(os.getenv("VIDEOSEEK_FRAME_SIZE", "512"))


def get_frame_quality() -> int:
    """JPEG compression quality (1–95) for frame thumbnails sent to the vision API."""
    return int(os.getenv("VIDEOSEEK_FRAME_QUALITY", "75"))
