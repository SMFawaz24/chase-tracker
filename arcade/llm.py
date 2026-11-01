"""Small, provider-agnostic LLM client for JSON tasks.

Every provider below speaks the OpenAI-compatible chat API, so one code path covers them all:

    groq    free tier   GROQ_API_KEY      https://console.groq.com
    gemini  free tier   GEMINI_API_KEY    https://aistudio.google.com/apikey
    openai  paid        OPENAI_API_KEY
    ollama  local, free (no key)          https://ollama.com

Responses are cached on disk (keyed by the full prompt), so re-running a build never pays twice and
works offline for anything already generated. If a provider rejects the default model name, pass
--model with a current one from that provider's model list.
"""
import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

PROVIDERS = {
    "groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.0-flash", "GEMINI_API_KEY"),
    "openai": ("https://api.openai.com/v1", "gpt-4o-mini", "OPENAI_API_KEY"),
    "ollama": ("http://localhost:11434/v1", "llama3.1", None),
}


class LLMError(RuntimeError):
    pass


def parse_json(text):
    """Pull a JSON value out of a model reply, tolerating code fences and chatter around it."""
    text = text.strip()
    for candidate in (text, *re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        i, j = text.find(open_c), text.rfind(close_c)
        if 0 <= i < j:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                pass
    raise LLMError("reply was not valid JSON")


class LLMClient:
    def __init__(self, provider, model=None, api_key=None, cache_path=None, delay=2.0, max_calls=200, timeout=60):
        if provider not in PROVIDERS:
            raise LLMError(f"unknown provider '{provider}' (choose from {', '.join(PROVIDERS)})")
        base, default_model, env = PROVIDERS[provider]
        self.provider, self.base_url, self.model = provider, base, model or default_model
        self.api_key = api_key or (os.environ.get(env) if env else None)
        if env and not self.api_key:
            raise LLMError(f"set {env} first (PowerShell: $env:{env}=\"your-key\")")
        self.delay, self.max_calls, self.timeout = delay, max_calls, timeout
        self.calls = 0
        self.cache_path = Path(cache_path) if cache_path else None
        self.cache = {}
        if self.cache_path and self.cache_path.exists():
            self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))

    def _key(self, system, user, temperature):
        raw = json.dumps([self.provider, self.model, system, user, temperature])
        return hashlib.sha256(raw.encode()).hexdigest()

    def json(self, system, user, temperature=0.4):
        key = self._key(system, user, temperature)
        if key in self.cache:
            return self.cache[key]
        if self.calls >= self.max_calls:
            raise LLMError("call budget reached (raise --max-calls)")
        reply = self._post(system, user, temperature)
        value = parse_json(reply)
        self.cache[key] = value
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self.cache), encoding="utf-8")
        return value

    def _post(self, system, user, temperature):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {"model": self.model, "temperature": temperature,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self.provider != "ollama":
            body["response_format"] = {"type": "json_object"}
        last = ""
        for attempt in range(4):
            time.sleep(self.delay if attempt == 0 else 5 * 2 ** attempt)
            self.calls += 1
            try:
                r = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=body, timeout=self.timeout)
            except requests.RequestException as e:
                last = str(e)
                continue
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            if r.status_code not in (429, 500, 502, 503, 504):
                break
        raise LLMError(last)


class MockLLM:
    """Test double: `handler(system, user)` returns the JSON value to use as the reply."""

    def __init__(self, handler):
        self.handler, self.calls = handler, 0

    def json(self, system, user, temperature=0.4):
        self.calls += 1
        return self.handler(system, user)
