"""Tests for configuration helpers."""

from __future__ import annotations

import os
import pytest

import videoseek_cli.config as cfg


class TestConfig:
    def test_mock_mode_when_no_key(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        assert cfg.is_mock_mode() is True

    def test_not_mock_mode_when_key_set(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-key")
        assert cfg.is_mock_mode() is False

    def test_default_model(self, monkeypatch):
        monkeypatch.delenv("VIDEOSEEK_MODEL", raising=False)
        assert cfg.get_model() == "google/gemini-2.5-flash"

    def test_custom_model(self, monkeypatch):
        monkeypatch.setenv("VIDEOSEEK_MODEL", "openai/gpt-5.4")
        assert cfg.get_model() == "openai/gpt-5.4"

    def test_default_fps(self, monkeypatch):
        monkeypatch.delenv("VIDEOSEEK_FPS", raising=False)
        assert cfg.get_fps() == 1.0

    def test_default_max_frames(self, monkeypatch):
        monkeypatch.delenv("VIDEOSEEK_MAX_FRAMES", raising=False)
        assert cfg.get_max_frames() == 300

    def test_default_threshold(self, monkeypatch):
        monkeypatch.delenv("VIDEOSEEK_SIMILARITY_THRESHOLD", raising=False)
        assert cfg.get_similarity_threshold() == 0.35

    def test_default_port(self, monkeypatch):
        monkeypatch.delenv("VIDEOSEEK_PORT", raising=False)
        assert cfg.get_port() == 8000

    def test_debug_false_by_default(self, monkeypatch):
        monkeypatch.delenv("VIDEOSEEK_DEBUG", raising=False)
        assert cfg.is_debug() is False

    def test_debug_true_when_set(self, monkeypatch):
        monkeypatch.setenv("VIDEOSEEK_DEBUG", "true")
        assert cfg.is_debug() is True
