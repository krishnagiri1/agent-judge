"""Thin, swappable LLM clients plus an on-disk response cache.

Providers
---------
openai     Any OpenAI-compatible endpoint (OpenAI, Groq, Together, OpenRouter, vLLM...).
           Uses OPENAI_API_KEY and optional --base-url / OPENAI_BASE_URL.
anthropic  Anthropic Messages API. Uses ANTHROPIC_API_KEY.
ollama     Local models via Ollama's native API (lets us raise the context window,
           which Ollama's OpenAI-compatible endpoint cannot). Free, runs on your GPU.
mock       Deterministic heuristic "judge" for tests and dry runs; no network, no cost.

Every call is cached by (provider, model, system, user), so re-running an experiment
or re-using a judgment in another experiment costs nothing.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import urllib.request
from pathlib import Path
from typing import Optional, Protocol


class LLMClient(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


def _with_retries(fn, attempts: int = 5, base_delay: float = 2.0):
    last = None
    for i in range(attempts):
        try:
            return fn()
        except Exception as e:  # rate limits, timeouts, transient 5xx
            last = e
            msg = str(e).lower()
            if any(k in msg for k in ("invalid api key", "authentication", "not found", "does not exist")):
                raise
            time.sleep(base_delay * (2**i))
    raise last  # type: ignore[misc]


class OpenAICompatClient:
    def __init__(self, model: str, base_url: Optional[str] = None, max_tokens: int = 1500):
        from openai import OpenAI

        self.model = model
        self.name = f"openai:{model}"
        self.max_tokens = max_tokens
        self.client = OpenAI(base_url=base_url or os.environ.get("OPENAI_BASE_URL") or None)
        self._kwargs = {"temperature": 0, "max_tokens": max_tokens}

    def complete(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        def call():
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, **self._kwargs)
            except Exception as e:
                msg = str(e)
                # Some newer models reject these parameters; adapt once and retry.
                if "max_tokens" in msg and "max_tokens" in self._kwargs:
                    self._kwargs.pop("max_tokens")
                    self._kwargs["max_completion_tokens"] = self.max_tokens
                    r = self.client.chat.completions.create(model=self.model, messages=messages, **self._kwargs)
                elif "temperature" in msg and "temperature" in self._kwargs:
                    self._kwargs.pop("temperature")
                    r = self.client.chat.completions.create(model=self.model, messages=messages, **self._kwargs)
                else:
                    raise
            return r.choices[0].message.content or ""

        return _with_retries(call)


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 1500):
        import anthropic

        self.model = model
        self.name = f"anthropic:{model}"
        self.max_tokens = max_tokens
        self.client = anthropic.Anthropic()
        self._temperature: Optional[float] = 0.0

    def complete(self, system: str, user: str) -> str:
        def call():
            kwargs = dict(model=self.model, max_tokens=self.max_tokens, system=system,
                          messages=[{"role": "user", "content": user}])
            if self._temperature is not None:
                kwargs["temperature"] = self._temperature
            try:
                r = self.client.messages.create(**kwargs)
            except Exception as e:
                if "temperature" in str(e) and self._temperature is not None:
                    self._temperature = None
                    kwargs.pop("temperature")
                    r = self.client.messages.create(**kwargs)
                else:
                    raise
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")

        return _with_retries(call)


class OllamaClient:
    def __init__(self, model: str, host: Optional[str] = None, num_ctx: int = 16384):
        self.model = model
        self.name = f"ollama:{model}"
        self.host = (host or os.environ.get("OLLAMA_HOST") or "http://localhost:11434").rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.num_ctx = num_ctx

    def complete(self, system: str, user: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
        }

        def call():
            req = urllib.request.Request(
                f"{self.host}/api/chat", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                return json.loads(resp.read())["message"]["content"]

        return _with_retries(call, attempts=3)


class MockClient:
    """Deterministic stand-in judge built from simple transcript heuristics.

    `position_bias` (0-1): share of pairwise calls that answer "A" no matter what.
    `verbosity_bias` (int): points added to reasoning/overall when agent turns are wordy.
    These knobs exist so the test suite can confirm the bias experiments detect bias.
    """

    def __init__(self, model: str = "heuristic", position_bias: float = 0.0, verbosity_bias: int = 0):
        self.model = model
        self.name = f"mock:{model}:pb{position_bias}:vb{verbosity_bias}"
        self.position_bias = position_bias
        self.verbosity_bias = verbosity_bias

    @staticmethod
    def _quality(transcript: str) -> float:
        patch = transcript.split("# FINAL PATCH", 1)[-1]
        q = 3.0
        q += 1.0 if "submit" in transcript[-6000:] else -1.0
        q -= 1.5 if "(no patch was produced)" in patch else 0.0
        q -= min(transcript.count("SyntaxError") + transcript.count("Traceback"), 6) * 0.25
        header = re.search(r"# AGENT TRAJECTORY \(.*?\)", transcript)
        q -= 0.5 if header and "exit_context" in header.group(0) else 0.0
        q += 0.5 if patch.count("\n+") < 40 else -0.5
        return max(1.0, min(5.0, q))

    @staticmethod
    def _avg_agent_turn_len(transcript: str) -> float:
        turns = re.findall(r"AGENT:\n(.*?)(?:\nOBSERVATION:|\n## Step|\Z)", transcript, flags=re.S)
        return sum(len(t) for t in turns) / max(len(turns), 1)

    def complete(self, system: str, user: str) -> str:
        h = int(hashlib.sha256(user.encode()).hexdigest(), 16)
        if "TRANSCRIPT A" in user and "TRANSCRIPT B" in user:
            a = user.split("TRANSCRIPT A", 1)[1].split("TRANSCRIPT B", 1)[0]
            b = user.split("TRANSCRIPT B", 1)[1]
            if (h % 1000) / 1000 < self.position_bias:
                winner = "A"
            else:
                qa, qb = self._quality(a), self._quality(b)
                winner = "A" if qa > qb else "B" if qb > qa else "tie"
            return json.dumps({"analysis": "heuristic comparison", "winner": winner, "confidence": 60})
        q = self._quality(user)
        bump = self.verbosity_bias if self._avg_agent_turn_len(user) > 250 else 0
        s = int(round(q))
        scores = {
            "correctness": s,
            "tool_use": max(1, min(5, s - (1 if user.count("Traceback") > 3 else 0))),
            "reasoning": max(1, min(5, s + bump)),
            "overall": max(1, min(5, s + bump)),
        }
        prob = int((q - 1) / 4 * 100)
        return json.dumps({"analysis": "heuristic", **scores, "resolved_probability": prob,
                           "rationale": "Heuristic score from submit status, errors and patch size."})


class CachedClient:
    """Wraps any client with a thread-safe SQLite cache keyed on the full request."""

    def __init__(self, inner: LLMClient, path: str | Path = ".cache/llm_cache.sqlite"):
        self.inner = inner
        self.name = inner.name
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS cache (k TEXT PRIMARY KEY, v TEXT)")
        self._db.commit()

    def _key(self, system: str, user: str) -> str:
        return hashlib.sha256(f"{self.inner.name}\x00{system}\x00{user}".encode()).hexdigest()

    def complete(self, system: str, user: str) -> str:
        k = self._key(system, user)
        with self._lock:
            row = self._db.execute("SELECT v FROM cache WHERE k=?", (k,)).fetchone()
        if row:
            return row[0]
        v = self.inner.complete(system, user)
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO cache VALUES (?, ?)", (k, v))
            self._db.commit()
        return v

    def forget(self, system: str, user: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM cache WHERE k=?", (self._key(system, user),))
            self._db.commit()


def make_client(
    provider: str,
    model: str,
    base_url: Optional[str] = None,
    cache: bool = True,
    cache_path: str | Path = ".cache/llm_cache.sqlite",
    **kwargs,
) -> LLMClient:
    provider = provider.lower()
    if provider == "openai":
        inner: LLMClient = OpenAICompatClient(model, base_url=base_url)
    elif provider == "anthropic":
        inner = AnthropicClient(model)
    elif provider == "ollama":
        inner = OllamaClient(model, host=base_url, **{k: v for k, v in kwargs.items() if k == "num_ctx"})
    elif provider == "mock":
        inner = MockClient(model, **{k: v for k, v in kwargs.items() if k in ("position_bias", "verbosity_bias")})
    else:
        raise ValueError(f"Unknown provider {provider!r}; use openai, anthropic, ollama or mock.")
    return CachedClient(inner, cache_path) if cache else inner
