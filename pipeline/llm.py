"""Minimal multi-provider LLM caller with disk cache and threaded map.

Frugality rules baked in:
- every completed call is cached on disk keyed by (provider, model, messages, params, seed);
  re-runs are free
- paid providers (openai, openrouter) log token usage to results/spend.jsonl so REPORT.md
  can account for every cent
- anthropic uses ANTHROPIC_API_KEY_LP (low-priority experiment lane) when available
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "llm"
SPEND_LOG = ROOT / "results" / "spend.jsonl"
_spend_lock = threading.Lock()


def load_env():
    env = {}
    for line in (ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if line.startswith("export ") and "=" in line:
            k, v = line[len("export "):].split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


_ENV = load_env()


def _key(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]


def _log_spend(provider: str, model: str, usage: dict):
    if provider == "anthropic":
        return  # free lane for us
    SPEND_LOG.parent.mkdir(parents=True, exist_ok=True)
    with _spend_lock:
        with open(SPEND_LOG, "a") as f:
            f.write(json.dumps({"t": time.time(), "provider": provider, "model": model,
                                "usage": usage}) + "\n")


def call(provider: str, model: str, system: str, user: str,
         temperature: float = 0.2, max_tokens: int = 512, seed: int = 0,
         timeout: float = 120.0) -> str:
    """One chat completion. `seed` only disambiguates the cache for repeated sampling."""
    payload = {"provider": provider, "model": model, "system": system, "user": user,
               "temperature": temperature, "max_tokens": max_tokens, "seed": seed}
    ck = _key(payload)
    cache_file = CACHE / ck[:2] / f"{ck}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["text"]

    last_err = None
    for attempt in range(6):
        try:
            if provider == "anthropic":
                api_key = os.environ.get("ANTHROPIC_API_KEY_LP") or _ENV.get("ANTHROPIC_API_KEY_LP") \
                    or os.environ.get("ANTHROPIC_API_KEY") or _ENV.get("ANTHROPIC_API_KEY")
                r = httpx.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={"x-api-key": api_key, "anthropic-version": "2023-06-01"},
                    json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
                          "system": system, "messages": [{"role": "user", "content": user}]},
                    timeout=timeout)
                r.raise_for_status()
                data = r.json()
                text = "".join(b.get("text", "") for b in data["content"])
                usage = data.get("usage", {})
            elif provider in ("openai", "openrouter"):
                if provider == "openai":
                    url = "https://api.openai.com/v1/chat/completions"
                    api_key = _ENV["OPENAI_API_KEY"]
                else:
                    url = "https://openrouter.ai/api/v1/chat/completions"
                    api_key = _ENV["OPENROUTER_KEY"]
                r = httpx.post(
                    url, headers={"Authorization": f"Bearer {api_key}"},
                    json={"model": model, "max_tokens": max_tokens, "temperature": temperature,
                          "messages": [{"role": "system", "content": system},
                                       {"role": "user", "content": user}]},
                    timeout=timeout)
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"] or ""
                usage = data.get("usage", {})
            else:
                raise ValueError(f"unknown provider {provider}")

            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps({"text": text, "usage": usage, "model": model}))
            _log_spend(provider, model, usage)
            return text
        except httpx.HTTPStatusError as e:
            last_err = e
            if e.response.status_code in (429, 500, 502, 503, 529):
                time.sleep(min(2 ** attempt * 2, 60))
                continue
            raise
        except (httpx.TimeoutException, httpx.TransportError) as e:
            last_err = e
            time.sleep(min(2 ** attempt * 2, 60))
    raise RuntimeError(f"exhausted retries: {last_err}")


def map_calls(jobs: list[dict], concurrency: int = 24) -> list[str | None]:
    """Threaded map over call() kwargs dicts; failed jobs return None (logged, not raised)."""
    out: list[str | None] = [None] * len(jobs)

    def run(i):
        try:
            out[i] = call(**jobs[i])
        except Exception as e:
            print(f"[job {i}] FAILED: {type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(run, range(len(jobs))))
    return out
