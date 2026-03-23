"""Tests for VisionClient retry logic and token tracking."""

from __future__ import annotations

import os

import pytest

from videoseek_cli.vision_client import TokenStats, VisionClient


# ---------------------------------------------------------------------------
# TokenStats unit tests
# ---------------------------------------------------------------------------

class TestTokenStats:
    def test_initial_state(self):
        ts = TokenStats()
        assert ts.prompt_tokens == 0
        assert ts.completion_tokens == 0
        assert ts.total_tokens == 0
        assert ts.api_calls == 0
        assert ts.retries == 0

    def test_add_accumulates(self):
        ts = TokenStats()
        ts.add(100, 50)
        assert ts.prompt_tokens == 100
        assert ts.completion_tokens == 50
        assert ts.total_tokens == 150
        assert ts.api_calls == 1

    def test_add_multiple_times(self):
        ts = TokenStats()
        ts.add(100, 50)
        ts.add(200, 80)
        assert ts.prompt_tokens == 300
        assert ts.completion_tokens == 130
        assert ts.total_tokens == 430
        assert ts.api_calls == 2

    def test_as_dict_keys(self):
        ts = TokenStats()
        ts.add(10, 5)
        d = ts.as_dict()
        assert "prompt_tokens" in d
        assert "completion_tokens" in d
        assert "total_tokens" in d
        assert "api_calls" in d
        assert "retries" in d

    def test_retries_not_incremented_by_add(self):
        ts = TokenStats()
        ts.add(10, 5)
        assert ts.retries == 0

    def test_retries_incremented_manually(self):
        ts = TokenStats()
        ts.retries += 1
        ts.retries += 1
        assert ts.retries == 2


# ---------------------------------------------------------------------------
# VisionClient mock mode tests
# ---------------------------------------------------------------------------

class TestVisionClientMockMode:
    def setup_method(self):
        os.environ["OPENROUTER_API_KEY"] = ""

    def test_describe_frame_returns_string_in_mock_mode(self):
        client = VisionClient()
        result = client.describe_frame("", "00:01:30")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_describe_frame_deterministic(self):
        client = VisionClient()
        r1 = client.describe_frame("", "00:01:30")
        r2 = client.describe_frame("", "00:01:30")
        assert r1 == r2

    def test_get_token_stats_returns_dict(self):
        client = VisionClient()
        stats = client.get_token_stats()
        assert isinstance(stats, dict)
        assert "total_tokens" in stats
        assert "api_calls" in stats

    def test_token_stats_zero_in_mock_mode(self):
        client = VisionClient()
        client.describe_frame("", "00:00:10")
        client.describe_frame("", "00:00:20")
        stats = client.get_token_stats()
        # In mock mode no API calls are made, so token usage stays zero
        assert stats["api_calls"] == 0
        assert stats["total_tokens"] == 0

    def test_batch_describe_returns_correct_count(self):
        client = VisionClient()
        frames = [
            {"index": i, "timestamp_seconds": float(i), "timestamp_str": f"00:00:0{i}", "image_b64": ""}
            for i in range(3)
        ]
        results = client.batch_describe(frames)
        assert len(results) == 3

    def test_batch_describe_calls_progress_callback(self):
        client = VisionClient()
        frames = [
            {"index": i, "timestamp_seconds": float(i), "timestamp_str": f"00:00:0{i}", "image_b64": ""}
            for i in range(4)
        ]
        calls = []
        client.batch_describe(frames, progress_callback=lambda done, total: calls.append((done, total)))
        assert len(calls) == 4
        assert calls[-1] == (4, 4)

    def test_model_override(self):
        client = VisionClient(model="openai/gpt-5.4-pro")
        assert client._model == "openai/gpt-5.4-pro"

    def test_default_model_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEOSEEK_MODEL", "x-ai/grok-4-fast")
        client = VisionClient()
        assert client._model == "x-ai/grok-4-fast"


# ---------------------------------------------------------------------------
# Retry config / wiring tests (mock mode, no real HTTP calls)
# ---------------------------------------------------------------------------

class TestRetryConfig:
    def test_retry_attempts_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEOSEEK_RETRY_ATTEMPTS", "5")
        client = VisionClient()
        assert client._retry_attempts == 5

    def test_retry_delay_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEOSEEK_RETRY_DELAY", "2.5")
        client = VisionClient()
        assert client._retry_delay == 2.5

    def test_max_tokens_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEOSEEK_MAX_TOKENS", "200")
        client = VisionClient()
        assert client._max_tokens == 200

    def test_vision_detail_from_env(self, monkeypatch):
        monkeypatch.setenv("VIDEOSEEK_VISION_DETAIL", "high")
        client = VisionClient()
        assert client._detail == "high"


# ---------------------------------------------------------------------------
# Retry behaviour test — patched HTTP client
# ---------------------------------------------------------------------------

class TestRetryBehaviour:
    """
    Test that VisionClient actually retries on transient network errors.
    Uses monkeypatching to avoid real HTTP calls.
    """

    def test_retries_on_timeout_then_succeeds(self, monkeypatch):
        import httpx
        from videoseek_cli import vision_client as vc_mod

        call_count = [0]

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "choices": [{"message": {"content": "A dog runs in the park."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }

        class FakeClient:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                call_count[0] += 1
                if call_count[0] < 2:
                    raise httpx.TimeoutException("timed out")
                return FakeResponse()

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr("httpx.Client", lambda **kw: FakeClient())
        # Suppress sleep so the test is fast
        monkeypatch.setattr("time.sleep", lambda s: None)

        client = VisionClient()
        client._retry_attempts = 3
        client._retry_delay = 0.01
        result = client.describe_frame("base64data==", "00:01:00")
        assert result == "A dog runs in the park."
        assert call_count[0] == 2
        assert client.token_stats.retries == 1

    def test_gives_up_after_max_retries(self, monkeypatch):
        import httpx

        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
        monkeypatch.setattr("time.sleep", lambda s: None)

        class AlwaysTimeout:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def post(self, url, **kwargs):
                raise httpx.TimeoutException("always times out")

        monkeypatch.setattr("httpx.Client", lambda **kw: AlwaysTimeout())

        client = VisionClient()
        client._retry_attempts = 2
        client._retry_delay = 0.0
        result = client.describe_frame("base64data==", "00:00:05")
        # Should return the fallback string, not raise
        assert "unavailable" in result.lower() or "00:00:05" in result
