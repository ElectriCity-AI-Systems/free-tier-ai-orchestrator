"""Thread-safe multi-provider AI HTTP client built on the standard library.

OpenRouter remains the zero-price default catalogue. Additional free-tier or
credit-backed providers are enabled automatically when their environment key is
present. The rest of the orchestrator sees one common chat/model interface.
"""
from __future__ import annotations

import json
import os
import random
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .config import (ANTHROPIC_BASE_URL, APP_REFERER, APP_TITLE,
                     FIREWORKS_BASE_URL, GEMINI_BASE_URL,
                     HUGGINGFACE_BASE_URL, KEYLESS_PROVIDERS,
                     MOONSHOT_BASE_URL, OPENAI_BASE_URL, OPENROUTER_BASE_URL,
                     PERPLEXITY_BASE_URL, PROVIDER_ENV, PROVIDER_KEY_ENVS,
                     PROVIDER_ORDER, REPLICATE_BASE_URL, Settings,
                     TOGETHER_BASE_URL)


# --------------------------------------------------------------------------- #
# Typed errors so callers can react intelligently (rotate model vs. abort).
# --------------------------------------------------------------------------- #
class OpenRouterError(Exception):
    """Base class for all provider client errors.

    The historical name is kept so existing callers/tests keep working.
    """


class AuthError(OpenRouterError):
    """401/403 - bad or missing API key. Fatal when no other provider works."""


class RateLimited(OpenRouterError):
    """429/402/quota - rotate to another model and cool this one down."""

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelUnavailable(OpenRouterError):
    """400/404/422 - this specific model is broken; try a different one."""


class TransientError(OpenRouterError):
    """5xx / network / timeout - retried with backoff on the same model."""


class BudgetExceeded(OpenRouterError):
    """Global request budget hit. Fatal; protects against runaway loops."""


PROVIDER_BASE_URLS = {
    "openrouter": OPENROUTER_BASE_URL,
    "openai": OPENAI_BASE_URL,
    "anthropic": ANTHROPIC_BASE_URL,
    "perplexity": PERPLEXITY_BASE_URL,
    "fireworks": FIREWORKS_BASE_URL,
    "together": TOGETHER_BASE_URL,
    "huggingface": HUGGINGFACE_BASE_URL,
    "gemini": GEMINI_BASE_URL,
    "moonshot": MOONSHOT_BASE_URL,
    "replicate": REPLICATE_BASE_URL,
    "oobabooga": os.environ.get(
        "OOBABOOGA_BASE_URL",
        os.environ.get("OOBA_BASE_URL", "http://127.0.0.1:5000/v1"),
    ).rstrip("/"),
}


OPENAI_COMPATIBLE = {
    "openrouter", "openai", "fireworks", "together", "huggingface",
    "oobabooga", "moonshot",
}


FALLBACK_MODELS: Dict[str, Sequence[Tuple[str, int, str]]] = {
    # Dynamic model discovery is preferred. These conservative fallbacks keep a
    # provider usable when its model-list endpoint is temporarily unavailable.
    "fireworks": (
        ("accounts/fireworks/models/kimi-k2-instruct-0905", 131072, "Kimi K2 Instruct"),
        ("accounts/fireworks/models/deepseek-r1", 160000, "DeepSeek R1"),
        ("accounts/fireworks/models/llama-v3p3-70b-instruct", 128000, "Llama 3.3 70B Instruct"),
        ("accounts/fireworks/models/qwen3-coder-480b-a35b-instruct", 128000, "Qwen3 Coder"),
        ("accounts/fireworks/models/llama-v3p1-8b-instruct", 128000, "Llama 3.1 8B Instruct"),
    ),
    "moonshot": (
        ("kimi-k2-0905-preview", 262144, "Kimi K2 0905"),
        ("kimi-k2-turbo-preview", 262144, "Kimi K2 Turbo"),
        ("moonshot-v1-128k", 131072, "Moonshot v1 128k"),
        ("moonshot-v1-32k", 32768, "Moonshot v1 32k"),
        ("kimi-latest", 131072, "Kimi Latest"),
    ),
    "openai": (
        ("gpt-5.5", 400000, "GPT-5.5"),
        ("gpt-5.5-codex", 400000, "GPT-5.5 Codex"),
        ("gpt-5.1", 400000, "GPT-5.1"),
        ("gpt-4.1", 1048576, "GPT-4.1"),
        ("gpt-4o", 128000, "GPT-4o"),
    ),
    "anthropic": (
        ("claude-opus-4-8", 200000, "Claude Opus 4.8"),
        ("claude-sonnet-4-6", 1000000, "Claude Sonnet 4.6"),
        ("claude-haiku-4-5", 200000, "Claude Haiku 4.5"),
        ("claude-3-5-sonnet-latest", 200000, "Claude 3.5 Sonnet"),
    ),
    "perplexity": (
        ("sonar", 128000, "Sonar"),
        ("sonar-pro", 200000, "Sonar Pro"),
        ("sonar-reasoning", 128000, "Sonar Reasoning"),
        ("sonar-reasoning-pro", 128000, "Sonar Reasoning Pro"),
        ("sonar-deep-research", 128000, "Sonar Deep Research"),
    ),
    "together": (
        ("deepseek-ai/DeepSeek-R1", 163840, "DeepSeek R1"),
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8", 131072, "Qwen3 Coder"),
        ("meta-llama/Llama-3.3-70B-Instruct-Turbo", 131072, "Llama 3.3 70B Turbo"),
        ("mistralai/Mistral-Small-24B-Instruct-2501", 32768, "Mistral Small"),
    ),
    "huggingface": (
        ("deepseek-ai/DeepSeek-R1", 163840, "DeepSeek R1"),
        ("openai/gpt-oss-120b", 131072, "GPT OSS 120B"),
        ("Qwen/Qwen3-Coder-480B-A35B-Instruct", 131072, "Qwen3 Coder"),
        ("meta-llama/Llama-3.3-70B-Instruct", 131072, "Llama 3.3 70B"),
    ),
    "gemini": (
        ("models/gemini-2.5-flash", 1048576, "Gemini 2.5 Flash"),
        ("models/gemini-2.5-flash-lite", 1048576, "Gemini 2.5 Flash Lite"),
        ("models/gemini-2.0-flash", 1048576, "Gemini 2.0 Flash"),
    ),
    "replicate": (
        ("deepseek-ai/deepseek-r1", 160000, "DeepSeek R1"),
        ("meta/meta-llama-3-70b-instruct", 8192, "Llama 3 70B Instruct"),
    ),
    "oobabooga": (
        ("local-model", 8192, "Oobabooga local model"),
    ),
}


def _to_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _to_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _backoff(attempt: int) -> float:
    """Exponential backoff with full jitter, capped."""
    return min(30.0, (2 ** attempt) * 0.5) * (0.5 + random.random())


def canonical_model_id(provider: str, upstream_id: str) -> str:
    return "%s:%s" % (provider, upstream_id)


def split_model_id(model_id: str) -> Tuple[str, str]:
    """Return (provider, upstream_model_id).

    Unprefixed ids are treated as OpenRouter for backwards compatibility with
    older leaderboards/tests and user-supplied model ids.
    """
    if ":" in model_id:
        provider, upstream = model_id.split(":", 1)
        if provider in PROVIDER_ORDER:
            return provider, upstream
    return "openrouter", model_id


def _csv_models(raw: str) -> List[Tuple[str, int, str]]:
    """Parse PROVIDER_MODELS overrides.

    Format: model,model|context,model|context|display name
    """
    out: List[Tuple[str, int, str]] = []
    for item in (raw or "").split(","):
        parts = [p.strip() for p in item.split("|")]
        if not parts or not parts[0]:
            continue
        mid = parts[0]
        ctx = _to_int(parts[1], 0) if len(parts) >= 2 else 0
        name = parts[2] if len(parts) >= 3 and parts[2] else mid
        out.append((mid, ctx, name))
    return out


def _env_models(provider: str) -> List[Tuple[str, int, str]]:
    return _csv_models(os.environ.get("%s_MODELS" % provider.upper(), ""))


def _normal_model(provider: str, upstream_id: str, name: str = "",
                  context_length: int = 0, pricing: Optional[dict] = None,
                  free_kind: str = "credits") -> dict:
    return {
        "id": canonical_model_id(provider, upstream_id),
        "provider": provider,
        "upstream_id": upstream_id,
        "name": name or upstream_id,
        "context_length": int(context_length or 0),
        "pricing": pricing or {},
        "free_tier": True,
        "free_kind": free_kind,
    }


def _fallback_catalogue(provider: str, free_kind: str = "credits") -> List[dict]:
    rows = _env_models(provider) or list(FALLBACK_MODELS.get(provider, ()))
    return [_normal_model(provider, mid, name, ctx, free_kind=free_kind)
            for mid, ctx, name in rows]


def _messages_to_prompt(messages: List[dict]) -> str:
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").upper()
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        parts.append("%s:\n%s" % (role, content.strip()))
    parts.append("ASSISTANT:")
    return "\n\n".join(parts)


def _compact_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_compact_text(v) for v in value)
    if isinstance(value, dict):
        for key in ("text", "content", "output", "message"):
            if key in value:
                return _compact_text(value[key])
        return json.dumps(value, ensure_ascii=False)
    return str(value)


# Substrings that mark a model as NOT a normal text-chat model (audio, image,
# video, music, embeddings, base-completion, realtime, deep-research, …). Keeps
# the chat orchestrator from picking models that would just fail or cost money.
_NON_CHAT_HINTS = (
    "embed", "rerank", "reranker", "whisper", "tts", "audio", "speech",
    "transcribe", "transcription", "translate", "translation", "realtime",
    "image", "dall-e", "vision", "clip", "sdxl", "stable-diffusion", "diffusion",
    "flux", "imagen", "sora", "veo", "lyria", "music", "video", "moderation",
    "guard", "ocr", "davinci", "babbage", "deep-research", "computer-use",
    "turbo-instruct", "upscal",
)


def _chatlike_model(model_id: str, meta: dict) -> bool:
    mid = (model_id or "").lower()
    task = str(meta.get("task") or meta.get("pipeline_tag") or meta.get("type")
               or meta.get("model_type") or "").lower()
    if any(x in mid for x in _NON_CHAT_HINTS):
        return False
    if task and not any(x in task for x in (
        "chat", "text", "language", "conversational", "generation", "completion"
    )):
        return False
    return True


def _zero_price_openrouter_model(model_id: str, meta: dict) -> bool:
    if model_id.endswith(":free"):
        return True
    pricing = meta.get("pricing", {}) or {}
    try:
        prompt = float(pricing.get("prompt", "0") or 0)
        completion = float(pricing.get("completion", "0") or 0)
    except (TypeError, ValueError):
        return False
    return prompt == 0.0 and completion == 0.0


def _best_context(meta: dict) -> int:
    candidates = [
        meta.get("context_length"),
        meta.get("contextLength"),
        meta.get("context_window"),
        meta.get("max_context_length"),
        meta.get("max_sequence_length"),
        meta.get("max_position_embeddings"),
        meta.get("max_model_len"),
        meta.get("input_token_limit"),
        meta.get("inputTokenLimit"),
        (meta.get("top_provider") or {}).get("context_length"),
    ]
    for val in candidates:
        ctx = _to_int(val, 0)
        if ctx:
            return ctx
    return 0


class RateLimiter:
    """Bounds concurrency and enforces a minimum spacing between requests."""

    def __init__(self, max_concurrency: int, min_interval: float):
        self._sem = threading.Semaphore(max(1, max_concurrency))
        self._min_interval = max(0.0, min_interval)
        self._lock = threading.Lock()
        self._last_start = 0.0

    def __enter__(self) -> "RateLimiter":
        self._sem.acquire()
        if self._min_interval > 0:
            with self._lock:
                now = time.monotonic()
                wait = self._min_interval - (now - self._last_start)
                if wait > 0:
                    time.sleep(wait)
                self._last_start = time.monotonic()
        return self

    def __exit__(self, *_exc) -> None:
        self._sem.release()


class ProviderAdapter:
    """Base adapter. Subclasses implement list_models and chat_once."""

    def __init__(self, owner: "OpenRouterClient", name: str, api_key: str,
                 base_url: str):
        self.owner = owner
        self.name = name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> dict:
        return {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

    def _json_request(self, url: str, method: str = "GET", body: Optional[dict] = None,
                      headers: Optional[dict] = None, timeout: Optional[float] = None) -> dict:
        payload = None if body is None else json.dumps(body).encode("utf-8")
        req_headers = dict(headers or self._headers())
        req = urllib.request.Request(url, data=payload, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.owner.s.request_timeout) as resp:
                raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as exc:
            raw_body = exc.read().decode("utf-8", "ignore")
            self._raise_http(exc.code, raw_body, exc.headers)
        except (urllib.error.URLError, socket.timeout) as exc:
            raise TransientError("Network error on %s: %s" % (self.name, exc))
        except ValueError as exc:
            raise TransientError("Invalid JSON from %s: %s" % (self.name, exc))
        raise TransientError("No response from %s" % self.name)

    def _raise_http(self, code: int, body: str, headers=None) -> None:
        msg = body[:500] or ("HTTP %d" % code)
        if code in (401, 403):
            raise AuthError("%s authentication failed: %s" % (self.name, msg))
        if code in (402, 429):
            retry_after = _to_float(headers.get("Retry-After") if headers else None)
            raise RateLimited("%s quota/rate limited: %s" % (self.name, msg), retry_after)
        if code in (400, 404, 422):
            raise ModelUnavailable("%s rejected the request (HTTP %d): %s"
                                   % (self.name, code, msg))
        if 500 <= code < 600:
            raise TransientError("HTTP %d from %s: %s" % (code, self.name, msg[:120]))
        raise OpenRouterError("HTTP %d from %s: %s" % (code, self.name, msg))

    def list_models(self) -> List[dict]:
        raise NotImplementedError

    def chat_once(self, upstream_model: str, messages: List[dict],
                  temperature: float, max_tokens: int) -> str:
        raise NotImplementedError


class OpenAICompatibleProvider(ProviderAdapter):
    """Adapter for /chat/completions style providers."""

    def __init__(self, owner: "OpenRouterClient", name: str, api_key: str,
                 base_url: str, free_kind: str = "credits"):
        super().__init__(owner, name, api_key, base_url)
        self.free_kind = free_kind

    def _headers(self) -> dict:
        h = super()._headers()
        if self.name == "openrouter":
            h["HTTP-Referer"] = APP_REFERER
            h["X-Title"] = APP_TITLE
        return h

    def _models_url(self) -> str:
        return self.base_url + "/models"

    def list_models(self) -> List[dict]:
        models: List[dict] = []
        try:
            data = self._json_request(self._models_url(), headers=self._headers())
            rows = data.get("data", data.get("models", [])) if isinstance(data, dict) else data
            if isinstance(rows, dict):
                rows = list(rows.values())
            if isinstance(rows, list):
                for item in rows:
                    if not isinstance(item, dict):
                        continue
                    mid = item.get("id") or item.get("name") or item.get("model")
                    if not mid or not _chatlike_model(str(mid), item):
                        continue
                    if self.name == "openrouter" and not _zero_price_openrouter_model(str(mid), item):
                        continue
                    pricing = item.get("pricing", {}) or {}
                    free_kind = self.free_kind
                    if self.name == "openrouter":
                        free_kind = "zero"
                    models.append(_normal_model(
                        self.name, str(mid),
                        item.get("display_name") or item.get("name") or str(mid),
                        _best_context(item), pricing=pricing, free_kind=free_kind,
                    ))
        except AuthError:
            raise
        except OpenRouterError as exc:
            self.owner.provider_errors[self.name] = str(exc)

        extra = _env_models(self.name)
        if extra:
            seen = {m["upstream_id"] for m in models}
            for mid, ctx, name in extra:
                if mid not in seen:
                    models.append(_normal_model(
                        self.name, mid, name, ctx,
                        free_kind=("zero" if self.name == "openrouter" else self.free_kind),
                    ))
        if not models and self.name != "openrouter":
            models = _fallback_catalogue(self.name, self.free_kind)
        return models[:self.owner.s.provider_model_limit]

    def chat_once(self, upstream_model: str, messages: List[dict],
                  temperature: float, max_tokens: int) -> str:
        body = {
            "model": upstream_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = self._json_request(
            self.base_url + "/chat/completions",
            method="POST", body=body, headers=self._headers(),
        )
        return self._parse_completion(canonical_model_id(self.name, upstream_model), data)

    @staticmethod
    def _parse_completion(model: str, data: dict) -> str:
        if isinstance(data, dict) and data.get("error"):
            err = data["error"]
            code = err.get("code")
            message = err.get("message", str(err))
            if code in (401, 403):
                raise AuthError(message)
            if code in (402, 429):
                raise RateLimited(message)
            if code in (400, 404, 422):
                raise ModelUnavailable(message)
            raise OpenRouterError(message)

        choices = (data or {}).get("choices") or []
        if not choices:
            raise TransientError("Empty response from %s" % model)
        message = choices[0].get("message", {}) or {}
        content = (message.get("content") or "").strip()
        if not content:
            content = (message.get("reasoning") or "").strip()
        if not content:
            raise TransientError("Blank content from %s" % model)
        return content


class FireworksProvider(OpenAICompatibleProvider):
    def _models_url(self) -> str:
        # Fireworks' public catalogue endpoint is account-scoped. The inference
        # API still accepts the returned accounts/.../models/... ids.
        root = self.base_url.replace("/inference/v1", "/v1")
        return root.rstrip("/") + "/accounts/fireworks/models?pageSize=200"

    def list_models(self) -> List[dict]:
        models = super().list_models()
        for m in models:
            if not m["upstream_id"].startswith("accounts/"):
                m["upstream_id"] = "accounts/fireworks/models/" + m["upstream_id"].split("/")[-1]
                m["id"] = canonical_model_id(self.name, m["upstream_id"])
        return models


class PerplexityProvider(OpenAICompatibleProvider):
    def list_models(self) -> List[dict]:
        # The Sonar chat API has a compact, curated model set. Prefer explicit
        # PERPLEXITY_MODELS overrides, then the bundled Sonar fallbacks.
        return _fallback_catalogue("perplexity", "credits")[:self.owner.s.provider_model_limit]


class AnthropicProvider(ProviderAdapter):
    def _headers(self) -> dict:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    def list_models(self) -> List[dict]:
        models: List[dict] = []
        try:
            data = self._json_request(self.base_url + "/models", headers=self._headers())
            rows = data.get("data", []) if isinstance(data, dict) else []
            for item in rows:
                if not isinstance(item, dict):
                    continue
                mid = item.get("id") or item.get("name")
                if not mid or not _chatlike_model(str(mid), item):
                    continue
                models.append(_normal_model(
                    "anthropic", str(mid),
                    item.get("display_name") or item.get("name") or str(mid),
                    _best_context(item), free_kind="credits",
                ))
        except AuthError:
            raise
        except OpenRouterError as exc:
            self.owner.provider_errors[self.name] = str(exc)

        extra = _env_models("anthropic")
        if extra:
            seen = {m["upstream_id"] for m in models}
            for mid, ctx, name in extra:
                if mid not in seen:
                    models.append(_normal_model(
                        "anthropic", mid, name, ctx, free_kind="credits"))
        if not models:
            models = _fallback_catalogue("anthropic", "credits")
        return models[:self.owner.s.provider_model_limit]

    @staticmethod
    def _normal_messages(messages: List[dict]) -> Tuple[str, List[dict]]:
        system_parts: List[str] = []
        out: List[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "system":
                system_parts.append(content)
                continue
            anth_role = "assistant" if role == "assistant" else "user"
            if out and out[-1]["role"] == anth_role:
                out[-1]["content"] += "\n\n" + content
            else:
                out.append({"role": anth_role, "content": content})
        if not out:
            out.append({"role": "user", "content": ""})
        return "\n\n".join(system_parts), out

    def chat_once(self, upstream_model: str, messages: List[dict],
                  temperature: float, max_tokens: int) -> str:
        system_text, anth_messages = self._normal_messages(messages)
        body = {
            "model": upstream_model,
            "max_tokens": max_tokens,
            "messages": anth_messages,
        }
        if system_text:
            body["system"] = system_text
        # Some newest Claude models reject non-default temperature; omit it for
        # Opus 4.x and otherwise pass the orchestrator setting through.
        if not upstream_model.lower().startswith("claude-opus-4-"):
            body["temperature"] = temperature

        data = self._json_request(
            self.base_url + "/messages", method="POST", body=body,
            headers=self._headers(),
        )
        if data.get("error"):
            err = data["error"]
            message = err.get("message", str(err))
            etype = str(err.get("type", "")).lower()
            if "authentication" in etype or "permission" in etype:
                raise AuthError(message)
            if "rate" in etype or "overload" in etype:
                raise RateLimited(message)
            if "invalid" in etype or "not_found" in etype:
                raise ModelUnavailable(message)
            raise OpenRouterError(message)
        parts = data.get("content") or []
        text = "".join((p.get("text") or "") for p in parts
                       if isinstance(p, dict) and p.get("type") == "text").strip()
        if not text:
            raise TransientError("Blank content from anthropic:%s" % upstream_model)
        return text


class GeminiProvider(ProviderAdapter):
    def _model_allowed(self, upstream_id: str, meta: dict) -> bool:
        mid = upstream_id.lower()
        methods = meta.get("supportedGenerationMethods") or []
        if "generateContent" not in methods:
            return False
        if "aqa" in mid or "-live" in mid or any(x in mid for x in _NON_CHAT_HINTS):
            return False
        # Gemini's free API tier is model-specific. Keep the default conservative
        # and route to Flash-family models unless the user overrides GEMINI_MODELS.
        return "flash" in mid

    def list_models(self) -> List[dict]:
        models: List[dict] = []
        try:
            url = self.base_url + "/models?" + urllib.parse.urlencode({"key": self.api_key})
            data = self._json_request(url, headers={"Content-Type": "application/json"})
            for item in data.get("models", []) if isinstance(data, dict) else []:
                if not isinstance(item, dict):
                    continue
                upstream = item.get("name", "")
                if not upstream or not self._model_allowed(upstream, item):
                    continue
                models.append(_normal_model(
                    "gemini", upstream,
                    item.get("displayName") or upstream,
                    _best_context(item),
                    free_kind="free_api",
                ))
        except AuthError:
            raise
        except OpenRouterError as exc:
            self.owner.provider_errors[self.name] = str(exc)

        extra = _env_models("gemini")
        if extra:
            seen = {m["upstream_id"] for m in models}
            for mid, ctx, name in extra:
                upstream = mid if mid.startswith("models/") else "models/" + mid
                if upstream not in seen:
                    models.append(_normal_model("gemini", upstream, name, ctx, free_kind="free_api"))
        if not models:
            models = _fallback_catalogue("gemini", "free_api")
        return models[:self.owner.s.provider_model_limit]

    def chat_once(self, upstream_model: str, messages: List[dict],
                  temperature: float, max_tokens: int) -> str:
        system_parts: List[dict] = []
        contents: List[dict] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False)
            if role == "system":
                system_parts.append({"text": content})
                continue
            gemini_role = "model" if role == "assistant" else "user"
            contents.append({"role": gemini_role, "parts": [{"text": content}]})
        if not contents:
            contents.append({"role": "user", "parts": [{"text": ""}]})

        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        if system_parts:
            body["systemInstruction"] = {"parts": system_parts}

        url = (self.base_url + "/" + upstream_model + ":generateContent?"
               + urllib.parse.urlencode({"key": self.api_key}))
        data = self._json_request(url, method="POST", body=body,
                                  headers={"Content-Type": "application/json"})
        if data.get("error"):
            err = data["error"]
            code = err.get("code")
            msg = err.get("message", str(err))
            if code in (401, 403):
                raise AuthError(msg)
            if code in (402, 429):
                raise RateLimited(msg)
            if code in (400, 404, 422):
                raise ModelUnavailable(msg)
            raise OpenRouterError(msg)
        candidates = data.get("candidates") or []
        if not candidates:
            raise TransientError("Empty response from %s" % upstream_model)
        parts = (((candidates[0].get("content") or {}).get("parts")) or [])
        text = "".join((p.get("text") or "") for p in parts if isinstance(p, dict)).strip()
        if not text:
            raise TransientError("Blank content from %s" % upstream_model)
        return text


class ReplicateProvider(ProviderAdapter):
    def list_models(self) -> List[dict]:
        # Replicate models expose heterogeneous input schemas, so catalogue
        # discovery is intentionally curated/overridable.
        return _fallback_catalogue("replicate", "credits")[:self.owner.s.provider_model_limit]

    def _prediction_url(self, upstream_model: str) -> str:
        owner, name = upstream_model.split("/", 1)
        return "%s/models/%s/%s/predictions" % (
            self.base_url, urllib.parse.quote(owner, safe=""),
            urllib.parse.quote(name, safe="")
        )

    def _prediction_headers(self) -> dict:
        h = self._headers()
        h["Prefer"] = "wait=%d" % max(1, min(60, int(self.owner.s.request_timeout)))
        return h

    def chat_once(self, upstream_model: str, messages: List[dict],
                  temperature: float, max_tokens: int) -> str:
        prompt = _messages_to_prompt(messages)
        variants = [
            {"prompt": prompt, "temperature": temperature, "max_tokens": max_tokens},
            {"prompt": prompt, "temperature": temperature, "max_new_tokens": max_tokens},
            {"prompt": prompt},
        ]
        last: Optional[OpenRouterError] = None
        for payload in variants:
            try:
                data = self._json_request(
                    self._prediction_url(upstream_model), method="POST",
                    body={"input": payload}, headers=self._prediction_headers(),
                )
                return self._wait_for_prediction(data, upstream_model)
            except ModelUnavailable as exc:
                last = exc
                continue
        raise last or ModelUnavailable("Replicate model unavailable: " + upstream_model)

    def _wait_for_prediction(self, data: dict, upstream_model: str) -> str:
        deadline = time.monotonic() + self.owner.s.request_timeout
        while True:
            status = (data or {}).get("status", "")
            if status == "succeeded":
                text = _compact_text(data.get("output")).strip()
                if not text:
                    raise TransientError("Blank output from %s" % upstream_model)
                return text
            if status in ("failed", "canceled"):
                raise ModelUnavailable("Replicate %s status for %s: %s"
                                       % (status, upstream_model, data.get("error", "")))
            get_url = ((data or {}).get("urls") or {}).get("get")
            if not get_url:
                raise TransientError("Replicate prediction did not return output yet.")
            if time.monotonic() >= deadline:
                raise TransientError("Timed out waiting for Replicate %s" % upstream_model)
            time.sleep(1.5)
            data = self._json_request(get_url, headers=self._headers())


class OpenRouterClient:
    """Synchronous, thread-safe multi-provider client.

    The class name stays OpenRouterClient for backwards compatibility with the
    rest of the package; it now aggregates all enabled provider adapters.
    """

    def __init__(self, settings: Settings,
                 on_request: Optional[Callable[[str, str], None]] = None):
        self.s = settings
        self.limiter = RateLimiter(settings.max_concurrency, settings.min_request_interval)
        self.on_request = on_request
        self._count_lock = threading.Lock()
        self.request_count = 0
        self.provider_errors: Dict[str, str] = {}
        self.adapters = self._build_adapters()

    def _build_adapters(self) -> Dict[str, ProviderAdapter]:
        keys = dict(self.s.provider_api_keys or {})
        if self.s.api_key:
            keys.setdefault("openrouter", self.s.api_key)

        raw_enabled = (self.s.enabled_providers or os.environ.get(PROVIDER_ENV, "")).strip()
        enabled = None
        if raw_enabled:
            enabled = {p.strip().lower() for p in raw_enabled.split(",") if p.strip()}

        adapters: Dict[str, ProviderAdapter] = {}
        for provider in PROVIDER_ORDER:
            if enabled is not None and provider not in enabled:
                continue
            key = keys.get(provider, "").strip()
            if not key:
                if enabled is not None and provider in KEYLESS_PROVIDERS:
                    key = os.environ.get(PROVIDER_KEY_ENVS.get(provider, ""), "").strip() or "local"
                else:
                    continue
            base_url = PROVIDER_BASE_URLS[provider]
            if provider == "fireworks":
                adapters[provider] = FireworksProvider(self, provider, key, base_url)
            elif provider == "anthropic":
                adapters[provider] = AnthropicProvider(self, provider, key, base_url)
            elif provider == "perplexity":
                adapters[provider] = PerplexityProvider(self, provider, key, base_url, "credits")
            elif provider == "gemini":
                adapters[provider] = GeminiProvider(self, provider, key, base_url)
            elif provider == "replicate":
                adapters[provider] = ReplicateProvider(self, provider, key, base_url)
            elif provider in OPENAI_COMPATIBLE:
                if provider == "openrouter":
                    free_kind = "zero"
                elif provider in KEYLESS_PROVIDERS:
                    free_kind = "local"
                else:
                    free_kind = "credits"
                adapters[provider] = OpenAICompatibleProvider(self, provider, key, base_url, free_kind)
        return adapters

    def enabled_provider_names(self) -> List[str]:
        return [p for p in PROVIDER_ORDER if p in self.adapters]

    def _bump_budget(self) -> None:
        with self._count_lock:
            if self.request_count >= self.s.max_total_requests:
                raise BudgetExceeded(
                    "Global request budget of %d reached - stopping to stay safe."
                    % self.s.max_total_requests
                )
            self.request_count += 1

    # -- model catalogue --------------------------------------------------- #
    def list_models(self) -> List[dict]:
        if not self.adapters:
            return []
        catalogue: List[dict] = []
        fatal_auth: List[str] = []
        for name in self.enabled_provider_names():
            adapter = self.adapters[name]
            try:
                catalogue.extend(adapter.list_models())
            except AuthError as exc:
                self.provider_errors[name] = str(exc)
                fatal_auth.append("%s: %s" % (name, str(exc)[:160]))
            except OpenRouterError as exc:
                self.provider_errors[name] = str(exc)
        if not catalogue and fatal_auth:
            raise AuthError("; ".join(fatal_auth))
        if not catalogue and self.provider_errors:
            raise OpenRouterError("; ".join("%s: %s" % (k, v[:160])
                                            for k, v in self.provider_errors.items()))
        return catalogue

    # -- chat completion --------------------------------------------------- #
    def chat(self, model: str, messages: List[dict],
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> str:
        """Call one model once (with internal retries for transient errors)."""
        self._bump_budget()
        provider, upstream_model = split_model_id(model)
        adapter = self.adapters.get(provider)
        if adapter is None:
            raise ModelUnavailable("Provider %s is not enabled for %s" % (provider, model))

        if self.on_request:
            try:
                self.on_request(model, messages[-1].get("content", "")[:80])
            except Exception:
                pass

        temp = self.s.temperature if temperature is None else temperature
        out_tokens = self.s.max_tokens if max_tokens is None else max_tokens
        last_transient: Optional[OpenRouterError] = None

        for attempt in range(self.s.max_retries + 1):
            try:
                with self.limiter:
                    return adapter.chat_once(upstream_model, messages, temp, out_tokens)
            except (AuthError, RateLimited, ModelUnavailable):
                raise
            except TransientError as exc:
                last_transient = exc
            if attempt < self.s.max_retries:
                time.sleep(_backoff(attempt))

        raise last_transient or TransientError("Exhausted retries calling %s" % model)
