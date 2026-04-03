"""Unified LLM client for Claude (mission planner) and Mistral (drone reasoning).

All LLM calls are async and non-blocking. If an API is unavailable or errors,
the system falls back to classical AI gracefully.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass

import httpx

logger = logging.getLogger(__name__)

# Cache for LLM responses to avoid redundant API calls
_response_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 30.0  # seconds


@dataclass
class LLMResponse:
    """Wrapper for an LLM response."""

    success: bool
    content: str = ""
    parsed: dict | None = None
    error: str | None = None
    source: str = ""  # "claude", "mistral", "cache", "fallback"
    latency_ms: float = 0.0


def _cache_key(model: str, prompt: str) -> str:
    """Generate a cache key from model + prompt."""
    return hashlib.md5(f"{model}:{prompt}".encode()).hexdigest()


def _get_cached(key: str) -> dict | None:
    """Return cached response if still valid."""
    if key in _response_cache:
        ts, data = _response_cache[key]
        if time.time() - ts < _CACHE_TTL:
            return data
        del _response_cache[key]
    return None


def _set_cached(key: str, data: dict) -> None:
    """Cache a response."""
    _response_cache[key] = (time.time(), data)
    # Evict old entries if cache grows too large
    if len(_response_cache) > 200:
        cutoff = time.time() - _CACHE_TTL
        stale = [k for k, (ts, _) in _response_cache.items() if ts < cutoff]
        for k in stale:
            del _response_cache[k]


async def call_claude(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> LLMResponse:
    """Call the Claude API for mission-level strategic reasoning.

    Uses the Anthropic SDK via HTTP to keep it simple and async.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return LLMResponse(success=False, error="ANTHROPIC_API_KEY not set", source="fallback")

    cache_key = _cache_key("claude", f"{system_prompt}:{user_prompt}")
    cached = _get_cached(cache_key)
    if cached is not None:
        return LLMResponse(
            success=True,
            content=cached.get("content", ""),
            parsed=cached,
            source="cache",
        )

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            resp.raise_for_status()
            data = resp.json()

        latency = (time.monotonic() - t0) * 1000
        content = data["content"][0]["text"]

        # Try to parse as JSON
        parsed = _try_parse_json(content)
        if parsed:
            _set_cached(cache_key, parsed)

        logger.info("Claude response in %.0fms", latency)
        return LLMResponse(
            success=True,
            content=content,
            parsed=parsed,
            source="claude",
            latency_ms=latency,
        )

    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        logger.warning("Claude API error (%.0fms): %s", latency, e)
        return LLMResponse(success=False, error=str(e), source="fallback", latency_ms=latency)


async def call_mistral(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.3,
) -> LLMResponse:
    """Call the Mistral API for drone-level tactical reasoning.

    Uses the Mistral free tier (pixtral/mistral-small).
    """
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    if not api_key:
        return LLMResponse(success=False, error="MISTRAL_API_KEY not set", source="fallback")

    cache_key = _cache_key("mistral", f"{system_prompt}:{user_prompt}")
    cached = _get_cached(cache_key)
    if cached is not None:
        return LLMResponse(
            success=True,
            content=cached.get("content", ""),
            parsed=cached,
            source="cache",
        )

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "mistral-small-latest",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "response_format": {"type": "json_object"},
                },
            )
            resp.raise_for_status()
            data = resp.json()

        latency = (time.monotonic() - t0) * 1000
        content = data["choices"][0]["message"]["content"]

        parsed = _try_parse_json(content)
        if parsed:
            _set_cached(cache_key, parsed)

        logger.info("Mistral response in %.0fms", latency)
        return LLMResponse(
            success=True,
            content=content,
            parsed=parsed,
            source="mistral",
            latency_ms=latency,
        )

    except Exception as e:
        latency = (time.monotonic() - t0) * 1000
        logger.warning("Mistral API error (%.0fms): %s", latency, e)
        return LLMResponse(success=False, error=str(e), source="fallback", latency_ms=latency)


def _try_parse_json(text: str) -> dict | None:
    """Attempt to parse a JSON object from LLM response text."""
    text = text.strip()
    # Handle markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        lines = [line for line in lines if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object within the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
    return None
