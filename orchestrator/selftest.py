"""Offline self-tests. No network and no API key required.

These exercise the parts that are easy to get subtly wrong (JSON extraction,
plan normalisation, model filtering/selection, model rotation) and run the
full orchestration pipeline against a deterministic fake client.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import tempfile
import threading
import wave

from . import agents
from .agent import Agent
from .client import (AnthropicProvider, AuthError, OpenAICompatibleProvider,
                     OpenRouterClient, OpenRouterError, RateLimited,
                     canonical_model_id, split_model_id)
from .config import Settings
from .godmode import GodMode, format_report
from .godmode_catalog import format_godmode_catalog
from .orchestrator import Orchestrator
from .registry import ModelRegistry
from .router import ModelRouter
from .stats import Leaderboard
from .tools import ToolBox


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
_FAKE_CATALOGUE = [
    {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1",
     "context_length": 64000, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "qwen/qwen-2.5-coder-32b-instruct:free", "name": "Qwen Coder",
     "context_length": 32000, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "meta-llama/llama-3.3-70b-instruct:free", "name": "Llama 3.3",
     "context_length": 128000, "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "google/gemini-2.0-flash-exp:free", "name": "Gemini Flash",
     "context_length": 1000000, "pricing": {"prompt": "0", "completion": "0"}},
    # A paid model that must be filtered out:
    {"id": "openai/gpt-4o", "name": "GPT-4o", "context_length": 128000,
     "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
]


class FakeClient:
    """Deterministic stand-in for OpenRouterClient used by the pipeline test."""

    def __init__(self):
        self.request_count = 0
        self._lock = threading.Lock()
        self.on_request = None

    def list_models(self):
        return list(_FAKE_CATALOGUE)

    def chat(self, model, messages, temperature=None, max_tokens=None):
        with self._lock:
            self.request_count += 1
        system = messages[0]["content"]
        if "lead planner" in system:
            return json.dumps({"subtasks": [
                {"id": "t1", "description": "Analyse the problem",
                 "role": "reasoning", "acceptance": "clear analysis", "depends_on": []},
                {"id": "t2", "description": "Write the answer",
                 "role": "general", "acceptance": "complete answer", "depends_on": ["t1"]},
            ]})
        if "rigorous, skeptical reviewer" in system:
            return json.dumps({"score": 90, "passed": True, "feedback": ""})
        if "expert judge" in system:
            return "MERGED: best-of solution from %s." % model
        if "integrator" in system:
            return "INTEGRATED DELIVERABLE covering all subtasks."
        if "verify whether a DELIVERABLE" in system:
            return json.dumps({"score": 95, "complete": True, "missing": ""})
        if system.startswith("Revise the DELIVERABLE"):
            return "REFINED DELIVERABLE."
        return "Worker output from %s." % model


class MultiProviderFakeClient(FakeClient):
    def list_models(self):
        return [
            {"id": canonical_model_id("gemini", "models/gemini-2.5-flash"),
             "provider": "gemini", "upstream_id": "models/gemini-2.5-flash",
             "name": "Gemini Flash", "context_length": 1000000,
             "free_tier": True, "free_kind": "free_api"},
            {"id": canonical_model_id("together", "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8"),
             "provider": "together", "upstream_id": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
             "name": "Qwen3 Coder", "context_length": 131072,
             "free_tier": True, "free_kind": "credits"},
            {"id": canonical_model_id("huggingface", "openai/gpt-oss-120b"),
             "provider": "huggingface", "upstream_id": "openai/gpt-oss-120b",
             "name": "GPT OSS", "context_length": 131072,
             "free_tier": True, "free_kind": "credits"},
        ]


class _RotatingClient(FakeClient):
    """Rate-limits the first model it sees, succeeds afterwards."""

    def __init__(self):
        super().__init__()
        self._first = None

    def chat(self, model, messages, temperature=None, max_tokens=None):
        if self._first is None:
            self._first = model
            raise RateLimited("simulated 429 on %s" % model, retry_after=0.01)
        if model == self._first:
            raise RateLimited("still limited", retry_after=0.01)
        return "ok from %s" % model


class _ScriptedClient(FakeClient):
    """Returns a fixed queue of responses regardless of model/prompt."""

    def __init__(self, responses):
        super().__init__()
        self._responses = list(responses)

    def chat(self, model, messages, temperature=None, max_tokens=None):
        with self._lock:
            self.request_count += 1
        if self._responses:
            return self._responses.pop(0)
        return json.dumps({"thought": "done", "tool": "finish",
                           "args": {"summary": "done"}})


class _FailProviderClient(FakeClient):
    """The top-ranked provider 404s; the router must fall through to a good one."""

    def list_models(self):
        return [
            {"id": canonical_model_id("fireworks", "accounts/fireworks/models/deepseek-r1"),
             "provider": "fireworks",
             "upstream_id": "accounts/fireworks/models/deepseek-r1",
             "name": "DeepSeek R1", "context_length": 163840,
             "free_tier": True, "free_kind": "credits"},
            {"id": "meta-llama/llama-3.3-70b-instruct:free", "provider": "openrouter",
             "upstream_id": "meta-llama/llama-3.3-70b-instruct:free",
             "name": "Llama 3.3", "context_length": 131000,
             "pricing": {"prompt": "0", "completion": "0"},
             "free_tier": True, "free_kind": "zero"},
        ]

    def chat(self, model, messages, temperature=None, max_tokens=None):
        with self._lock:
            self.request_count += 1
        if model.startswith("fireworks:"):
            from .client import ModelUnavailable
            raise ModelUnavailable("404 Model not found: " + model)
        return "ok from " + model


class _GodModeAllClient(FakeClient):
    def list_models(self):
        return [
            {"id": "plain/general-chat:free", "name": "General Chat",
             "context_length": 8000, "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "coder/qwen-coder:free", "name": "Coder",
             "context_length": 8000, "pricing": {"prompt": "0", "completion": "0"}},
            {"id": "reason/deepseek-r1:free", "name": "Reasoner",
             "context_length": 8000, "pricing": {"prompt": "0", "completion": "0"}},
        ]

    def chat(self, model, messages, temperature=None, max_tokens=None):
        with self._lock:
            self.request_count += 1
        return "ok from " + model


class _AuthFailThenOkClient(FakeClient):
    def list_models(self):
        return [
            {"id": canonical_model_id("huggingface", "deepseek-ai/DeepSeek-R1"),
             "provider": "huggingface", "upstream_id": "deepseek-ai/DeepSeek-R1",
             "name": "Bad Auth", "context_length": 8000,
             "free_tier": True, "free_kind": "credits"},
            {"id": canonical_model_id("huggingface", "Qwen/Qwen3-Coder"),
             "provider": "huggingface", "upstream_id": "Qwen/Qwen3-Coder",
             "name": "Bad Auth 2", "context_length": 8000,
             "free_tier": True, "free_kind": "credits"},
            {"id": "openrouter/good-chat:free", "provider": "openrouter",
             "upstream_id": "openrouter/good-chat:free",
             "name": "Good", "context_length": 8000,
             "pricing": {"prompt": "0", "completion": "0"},
             "free_tier": True, "free_kind": "zero"},
        ]

    def chat(self, model, messages, temperature=None, max_tokens=None):
        with self._lock:
            self.request_count += 1
        if model.startswith("huggingface:"):
            raise AuthError("simulated bad Hugging Face token")
        return "ok from " + model


class _SilentUI:
    """No-op UI so tests stay quiet."""

    def __getattr__(self, _name):
        def _noop(*_a, **_k):
            return None
        return _noop

    def confirm(self, *_a, **_k):
        return False


class _ProviderOwner:
    def __init__(self):
        self.s = Settings(api_key="x", min_request_interval=0.0)
        self.provider_errors = {}


class _OpenRouterCatalogueAdapter(OpenAICompatibleProvider):
    def _json_request(self, *_a, **_k):
        return {"data": [
            {"id": "openai/gpt-4o", "name": "GPT-4o",
             "pricing": {"prompt": "0.0000025", "completion": "0.00001"}},
            {"id": "deepseek/deepseek-r1:free", "name": "DeepSeek R1",
             "context_length": 64000,
             "pricing": {"prompt": "0", "completion": "0"}},
        ]}


class _OpenAICatalogueAdapter(OpenAICompatibleProvider):
    def _json_request(self, *_a, **_k):
        return {"data": [
            {"id": "gpt-5.5", "name": "GPT-5.5",
             "context_length": 400000},
            {"id": "text-embedding-3-large", "name": "embedding"},
        ]}


class _AnthropicAdapter(AnthropicProvider):
    def _json_request(self, url, method="GET", body=None, **_k):
        if method == "GET" and url.endswith("/models"):
            return {"data": [
                {"id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6"},
            ]}
        assert method == "POST" and url.endswith("/messages"), url
        assert body["system"] == "sys", body
        assert body["messages"] == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ], body
        return {"content": [{"type": "text", "text": "anthropic ok"}]}


# --------------------------------------------------------------------------- #
# Individual checks
# --------------------------------------------------------------------------- #
def _check_json_extraction():
    cases = [
        ('{"a": 1}', {"a": 1}),
        ('here you go:\n```json\n{"a": 2}\n```\nthanks', {"a": 2}),
        ('prefix [1, 2, 3] suffix', [1, 2, 3]),
        ('noise {"nested": {"x": [1,2]}} more noise', {"nested": {"x": [1, 2]}}),
        ('text with "quoted } brace" {"ok": true} end', {"ok": True}),
    ]
    for text, expected in cases:
        got = agents.extract_json(text)
        assert got == expected, "extract_json(%r) -> %r != %r" % (text, got, expected)
    assert agents.extract_json("no json here") is None


def _check_plan_parsing():
    plan = agents.parse_plan('{"subtasks":[{"description":"do x","role":"coder"},'
                             '{"id":"t2","description":"do y","depends_on":["t1","ghost"]}]}')
    assert plan and len(plan) == 2, plan
    assert plan[0]["id"] == "t1" and plan[0]["role"] == "coder"
    # the unknown "ghost" dependency must be dropped:
    assert plan[1]["depends_on"] == ["t1"], plan[1]
    assert agents.parse_plan("garbage") is None


def _check_verdict_parsing():
    score, passed, _ = agents.parse_verdict('{"score": 85, "passed": true}', threshold=80)
    assert score == 85 and passed is True
    # explicit pass but below threshold must NOT pass:
    score, passed, _ = agents.parse_verdict('{"score": 70, "passed": true}', threshold=80)
    assert passed is False, (score, passed)


def _check_registry():
    reg = ModelRegistry(Settings(api_key="x"))
    models = reg.load(FakeClient())
    ids = {m.id for m in models}
    assert "openai/gpt-4o" not in ids, "paid model leaked into free registry"
    assert len(models) == 4, ids
    coders = reg.select("coder", n=2)
    assert any("coder" in m.id for m in coders), [m.id for m in coders]
    # diversity: 3 picks should span >=2 vendors
    picks = reg.select("general", n=3)
    assert len({m.vendor for m in picks}) >= 2, [m.id for m in picks]
    # cooldown removes a model from selection
    top = reg.select("general", n=1)[0]
    reg.penalize(top.id, retry_after=999)
    assert reg.select("general", n=1)[0].id != top.id


def _check_multi_provider_registry():
    assert split_model_id("gemini:models/gemini-2.5-flash") == (
        "gemini", "models/gemini-2.5-flash")
    assert split_model_id("anthropic:claude-sonnet-4-6") == (
        "anthropic", "claude-sonnet-4-6")
    # Historical OpenRouter ids can contain :free and must remain unprefixed.
    assert split_model_id("deepseek/deepseek-r1:free") == (
        "openrouter", "deepseek/deepseek-r1:free")

    reg = ModelRegistry(Settings(api_key="x"))
    reg.load(MultiProviderFakeClient())
    picks = reg.select("reasoning", n=3)
    assert len({m.provider for m in picks}) >= 2, [m.id for m in picks]
    assert any(m.provider == "gemini" for m in picks), [m.id for m in picks]
    coders = reg.select("coder", n=1)
    assert coders and "qwen" in coders[0].upstream_id.lower(), coders


def _check_openrouter_catalogue_filter():
    adapter = _OpenRouterCatalogueAdapter(_ProviderOwner(), "openrouter", "x",
                                          "https://example.test")
    models = adapter.list_models()
    ids = {m["upstream_id"] for m in models}
    assert "deepseek/deepseek-r1:free" in ids, ids
    assert "openai/gpt-4o" not in ids, ids


def _check_godmode_api_provider_adapters():
    openai = _OpenAICatalogueAdapter(_ProviderOwner(), "openai", "x",
                                     "https://example.test", "credits")
    models = openai.list_models()
    ids = {m["upstream_id"] for m in models}
    assert "gpt-5.5" in ids, ids
    assert "text-embedding-3-large" not in ids, ids

    anthropic = _AnthropicAdapter(_ProviderOwner(), "anthropic", "x",
                                  "https://example.test")
    models = anthropic.list_models()
    assert models and models[0]["provider"] == "anthropic", models
    text = anthropic.chat_once("claude-sonnet-4-6", [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ], temperature=0.2, max_tokens=100)
    assert text == "anthropic ok", text

    local = OpenRouterClient(Settings(enabled_providers="oobabooga",
                                      min_request_interval=0.0))
    assert local.enabled_provider_names() == ["oobabooga"], local.enabled_provider_names()

    px = OpenRouterClient(Settings(provider_api_keys={"perplexity": "x"},
                                   enabled_providers="perplexity",
                                   min_request_interval=0.0))
    sonar = px.list_models()
    assert any(m["upstream_id"] == "sonar" for m in sonar), sonar


def _check_godmode_provider_catalog():
    text = format_godmode_catalog()
    assert "ChatGPT" in text and "openai" in text, text
    assert "Claude2" in text and "anthropic" in text, text
    assert "Bing" in text and "webapp-only" in text, text


def _check_model_rotation():
    settings = Settings(api_key="x", min_request_interval=0.0)
    reg = ModelRegistry(settings)
    reg.load(FakeClient())
    orch = Orchestrator(settings, _RotatingClient(), reg, _SilentUI())
    text, model = orch._call_role("general", [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    assert text.startswith("ok from"), (text, model)


def _check_tools():
    root = tempfile.mkdtemp(prefix="ofo_tools_")
    try:
        tb = ToolBox(Settings(api_key="x", max_tool_output=20), _SilentUI(),
                     root=root, auto_approve=True, allow_bash=False)
        assert "run_bash" not in tb.names(), "bash should be disabled"
        assert "consult_models" not in tb.names(), "consult tool needs a callback"
        assert "master_audio" in tb.names(), "audio mastering tool should be available"

        r = tb.run("write_file", {"path": "a.txt", "content": "hello world"})
        assert r.ok and os.path.isfile(os.path.join(root, "a.txt")), r.output

        r = tb.run("read_file", {"path": "a.txt"})
        assert r.ok and "hello" in r.output, r.output

        r = tb.run("list_dir", {"path": "."})
        assert "a.txt" in r.output, r.output

        r = tb.run("edit_file", {"path": "a.txt", "find": "hello", "replace": "bye"})
        assert r.ok, r.output
        with open(os.path.join(root, "a.txt")) as fh:
            assert fh.read() == "bye world"

        # sandbox: escaping the root must be refused
        r = tb.run("read_file", {"path": "../../etc/hostname"})
        assert not r.ok and "escape" in r.output, r.output

        # disabled bash returns an error rather than executing
        r = tb.run("run_bash", {"command": "echo hi"})
        assert not r.ok and "disabled" in r.output, r.output

        r = tb.run("master_audio", {"path": "missing.wav"})
        assert not r.ok and "no such audio file" in r.output, r.output

        # truncation honours max_tool_output
        tb.run("write_file", {"path": "big.txt", "content": "x" * 200})
        r = tb.run("read_file", {"path": "big.txt"})
        assert "truncated" in r.output, r.output
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _check_consult_models_tool():
    root = tempfile.mkdtemp(prefix="ofo_consult_")
    calls = []

    def fake_consult(prompt, role="reasoning", max_models=None):
        calls.append((prompt, role, max_models))
        return "consensus for: " + prompt

    try:
        tb = ToolBox(Settings(api_key="x"), _SilentUI(), root=root,
                     auto_approve=True, allow_bash=False,
                     model_consultant=fake_consult)
        assert "consult_models" in tb.names(), tb.names()
        assert "consult_models" in tb.spec(), tb.spec()
        assert "run_bash" not in tb.spec(), tb.spec()
        r = tb.run("consult_models", {
            "prompt": "choose mastering parameters",
            "max_models": 7,
        })
        assert r.ok and "consensus" in r.output, r.output
        assert calls == [("choose mastering parameters", "all", 7)], calls
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _check_godmode_diversity():
    from .godmode import GodMode
    settings = Settings(api_key="x", min_request_interval=0.0)
    client = FakeClient()
    reg = ModelRegistry(settings)
    reg.load(client)
    gm = GodMode(settings, ModelRouter(settings, client, reg), reg)
    picks = gm._models_for("all", 3)
    assert len(picks) == 3, [m.id for m in picks]
    assert len({m.vendor for m in picks}) == 3, "panel must be diverse vendors"
    res = gm.run("hello", role="all", width=2, synthesize=True)
    assert len(res.answers) == 2, res.answers


def _check_consult_default_cap():
    from .cli import _build_model_consultant
    settings = Settings(api_key="x", min_request_interval=0.0,
                        consult_default_models=2)
    client = FakeClient()
    reg = ModelRegistry(settings)
    reg.load(client)
    router = ModelRouter(settings, client, reg)
    consult = _build_model_consultant(settings, router, reg, _SilentUI())
    # No max_models -> must NOT consult all 4; capped to the default of 2.
    out = consult("please advise", role="all", max_models=None)
    assert "consulted 2 model(s)" in out, out
    # Explicit max_models widens it deliberately.
    out2 = consult("again", role="all", max_models=3)
    assert "consulted 3 model(s)" in out2, out2


def _check_moonshot_provider():
    from .client import PROVIDER_BASE_URLS, OPENAI_COMPATIBLE, OpenRouterClient
    from .config import PROVIDER_ORDER, PROVIDER_KEY_ENVS
    assert "moonshot" in PROVIDER_ORDER, PROVIDER_ORDER
    assert PROVIDER_BASE_URLS.get("moonshot"), "moonshot base url missing"
    assert "moonshot" in OPENAI_COMPATIBLE, "moonshot should be OpenAI-compatible"
    assert PROVIDER_KEY_ENVS["moonshot"] == "MOONSHOT_API_KEY"
    client = OpenRouterClient(Settings(api_key="", provider_api_keys={"moonshot": "sk-x"}))
    assert "moonshot" in client.enabled_provider_names(), client.enabled_provider_names()


def _check_login_env_update():
    from . import login
    root = tempfile.mkdtemp(prefix="ofo_login_")
    old = os.environ.get("XDG_CONFIG_HOME")
    try:
        os.environ["XDG_CONFIG_HOME"] = root
        # the user's four resolve to canonical provider ids
        assert login.resolve(["openai", "claude", "gemini", "kimi"]) == \
            ["openai", "anthropic", "gemini", "moonshot"]
        login.update_env({"OPENAI_API_KEY": "sk-a", "MOONSHOT_API_KEY": "sk-b"})
        login.update_env({"OPENAI_API_KEY": "sk-a2", "GEMINI_API_KEY": "AIza-x"})
        content = open(login.env_path()).read()
        assert "OPENAI_API_KEY=sk-a2" in content, content
        assert "OPENAI_API_KEY=sk-a\n" not in content, "old value not replaced cleanly"
        assert "MOONSHOT_API_KEY=sk-b" in content, "existing key must be preserved"
        assert "GEMINI_API_KEY=AIza-x" in content, "new key must be added"
        login.remove_env(["OPENAI_API_KEY"])
        content = open(login.env_path()).read()
        assert "OPENAI_API_KEY" not in content and "MOONSHOT_API_KEY=sk-b" in content
    finally:
        if old is None:
            os.environ.pop("XDG_CONFIG_HOME", None)
        else:
            os.environ["XDG_CONFIG_HOME"] = old
        shutil.rmtree(root, ignore_errors=True)


def _check_temperature_adaptation():
    from .client import ModelUnavailable, OpenAICompatibleProvider
    owner = _ProviderOwner()

    class _TempPicky(OpenAICompatibleProvider):
        seen = []
        def _json_request(self, url, method="GET", body=None, headers=None, timeout=None):
            _TempPicky.seen.append(dict(body or {}))
            if "temperature" in (body or {}):
                raise ModelUnavailable("invalid temperature: only 1 is allowed for this model")
            return {"choices": [{"message": {"content": "ok"}}]}

    p = _TempPicky(owner, "moonshot", "k", "https://api.example/v1")
    assert p.chat_once("kimi-k2.7", [{"role": "user", "content": "hi"}], 0.3, 100) == "ok"
    assert len(_TempPicky.seen) == 2 and "temperature" not in _TempPicky.seen[1]

    class _MaxTokPicky(OpenAICompatibleProvider):
        seen = []
        def _json_request(self, url, method="GET", body=None, headers=None, timeout=None):
            _MaxTokPicky.seen.append(dict(body or {}))
            if "max_tokens" in (body or {}):
                raise ModelUnavailable("Unsupported parameter: 'max_tokens' is not "
                                       "supported; use 'max_completion_tokens'.")
            return {"choices": [{"message": {"content": "ok2"}}]}

    q = _MaxTokPicky(owner, "openai", "k", "https://api.example/v1")
    assert q.chat_once("o3", [{"role": "user", "content": "hi"}], 0.3, 100) == "ok2"
    assert "max_completion_tokens" in _MaxTokPicky.seen[-1] \
        and "max_tokens" not in _MaxTokPicky.seen[-1]


def _check_non_chat_filter():
    from .client import _chatlike_model
    for bad in ("sora-2", "gpt-4o-transcribe", "gpt-realtime", "davinci-002",
                "babbage-002", "o3-deep-research", "gemini-2.5-flash-image",
                "google/lyria-3-pro-preview", "gpt-3.5-turbo-instruct",
                "text-embedding-3-large", "dall-e-3", "whisper-1"):
        assert not _chatlike_model(bad, {}), "must reject non-chat model: " + bad
    for good in ("gpt-5.5", "claude-opus-4-8", "qwen/qwen3-coder", "gpt-4o-mini",
                 "kimi-k2.7-code", "moonshot-v1-128k", "deepseek-r1",
                 "gpt-4o-search-preview", "meta-llama/llama-3.3-70b-instruct"):
        assert _chatlike_model(good, {}), "must accept chat model: " + good


def _check_free_only():
    catalogue = [
        {"id": "deepseek/deepseek-r1:free", "provider": "openrouter",
         "upstream_id": "deepseek/deepseek-r1:free", "free_tier": True,
         "free_kind": "zero", "context_length": 64000,
         "pricing": {"prompt": "0", "completion": "0"}},
        {"id": "gemini:models/gemini-2.5-flash", "provider": "gemini",
         "upstream_id": "models/gemini-2.5-flash", "free_tier": True,
         "free_kind": "free_api", "context_length": 1000000},
        {"id": "openai:gpt-5.5", "provider": "openai", "upstream_id": "gpt-5.5",
         "free_tier": True, "free_kind": "credits", "context_length": 0},
    ]

    class _Cat:
        def list_models(self):
            return list(catalogue)

    reg = ModelRegistry(Settings(api_key="x", free_only=True))
    reg.load(_Cat())
    kinds = {m.free_kind for m in reg.models}
    assert "credits" not in kinds, [m.id for m in reg.models]
    assert "zero" in kinds and "free_api" in kinds, kinds
    reg2 = ModelRegistry(Settings(api_key="x", free_only=False))
    reg2.load(_Cat())
    assert any(m.free_kind == "credits" for m in reg2.models), "paid kept by default"


def _check_web_events():
    from .web import EventBus, WebUI
    bus = EventBus()
    sub = bus.subscribe()
    bus.publish({"type": "x", "n": 1})
    assert sub.get_nowait()["n"] == 1
    ui = WebUI(bus)
    ui.agent_step(1, "m", "thinking", "read_file", {"path": "a"})
    replay = bus.subscribe()                     # fresh tab replays history
    seen = []
    while not replay.empty():
        seen.append(replay.get_nowait())
    assert any(e.get("type") == "step" and e.get("tool") == "read_file"
               for e in seen), seen

    # approval blocks until the browser answers
    sub3 = bus.subscribe()
    result = {}
    t = threading.Thread(target=lambda: result.__setitem__("r", ui.confirm("do X?")))
    t.start()
    aid = None
    for _ in range(8):
        ev = sub3.get(timeout=2)
        if ev.get("type") == "approval":
            aid = ev["id"]
            break
    assert aid is not None, "no approval event emitted"
    assert ui.resolve_approval(aid, True)
    t.join(timeout=2)
    assert result.get("r") is True
    assert ui.confirm("auto", assume_yes=True) is True


def _check_web_server():
    import urllib.request
    from .web import build_app, serve
    settings = Settings(api_key="x", min_request_interval=0.0)
    reg = ModelRegistry(settings)
    reg.load(FakeClient())
    try:
        app = build_app(settings, FakeClient(), reg, os.getcwd())
        httpd = serve(app, host="127.0.0.1", port=0, open_browser=False)
    except OSError:
        return  # environment can't bind a socket; skip rather than fail
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        port = httpd.server_address[1]
        base = "http://127.0.0.1:%d" % port
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "Free-Tier AI" in html and "EventSource" in html, "index html broken"
        data = json.loads(urllib.request.urlopen(base + "/api/models", timeout=5).read())
        assert data["count"] >= 1, data
        assert "providers" in data and "routing" in data, data
    finally:
        httpd.shutdown()


def _check_pro_licenses():
    from .pro import run_self_tests
    assert run_self_tests()


def _check_audio_mastering_tool():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        return
    root = tempfile.mkdtemp(prefix="ofo_audio_")
    try:
        src = os.path.join(root, "tone.wav")
        rate = 44100
        with wave.open(src, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            frames = []
            for n in range(rate):
                val = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * n / rate))
                frames.append(struct.pack("<h", val))
            wf.writeframes(b"".join(frames))

        tb = ToolBox(Settings(api_key="x", max_tool_output=2000), _SilentUI(),
                     root=root, auto_approve=True, allow_bash=False)
        r = tb.run("master_audio", {
            "path": "tone.wav",
            "output": "tone_tunecore_master.wav",
            "profile": "tunecore",
        })
        assert r.ok, r.output
        out = os.path.join(root, "tone_tunecore_master.wav")
        assert os.path.isfile(out) and os.path.getsize(out) > 1000, r.output
        assert tb.undo().ok
        assert not os.path.exists(out), "undo should remove created master"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _check_agent_loop():
    root = tempfile.mkdtemp(prefix="ofo_agent_")
    try:
        settings = Settings(api_key="x", min_request_interval=0.0,
                            verify_finish=False, max_steps=5)
        client = _ScriptedClient([
            json.dumps({"thought": "create file", "tool": "write_file",
                        "args": {"path": "greeting.txt", "content": "hello"}}),
            json.dumps({"thought": "done", "tool": "finish",
                        "args": {"summary": "created greeting.txt"}}),
        ])
        reg = ModelRegistry(settings)
        reg.load(client)
        router = ModelRouter(settings, client, reg)
        tb = ToolBox(settings, _SilentUI(), root=root,
                     auto_approve=True, allow_bash=False)
        agent = Agent(settings, router, _SilentUI(), tb, root=root)
        summary = agent.handle("create greeting.txt containing hello")
        assert "created" in summary.lower(), summary
        path = os.path.join(root, "greeting.txt")
        assert os.path.isfile(path), "agent did not create the file"
        with open(path) as fh:
            assert fh.read() == "hello"
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _check_full_pipeline():
    settings = Settings(api_key="x", min_request_interval=0.0,
                        ensemble_size=2, max_task_attempts=2, refine_passes=0)
    reg = ModelRegistry(settings)
    reg.load(FakeClient())
    orch = Orchestrator(settings, FakeClient(), reg, _SilentUI())
    result = orch.run("Explain something complex and write it up.")
    assert result["deliverable"], "empty deliverable"
    assert result["score"] == 95, result["score"]
    assert result["status"] == "GOAL ACHIEVED", result["status"]
    assert result["requests"] > 0
    assert len(result["plan"]) == 2


def _check_lenient_json():
    # Literal newline inside a JSON string (what models emit for file content)
    # must parse - this was the #1 cause of agent stalls.
    txt = '{"tool": "write_file", "args": {"path": "x.py", "content": "a\nb"}}'
    obj = agents.extract_json(txt)
    assert obj and obj["args"]["content"] == "a\nb", obj
    # trailing-comma repair
    assert agents.extract_json('{"a": 1,}') == {"a": 1}
    # fenced + prose around it
    assert agents.extract_json('sure:\n```json\n{"ok": true}\n```') == {"ok": True}


def _check_normalize_action():
    a = agents.normalize_action({"tool": "read_file", "args": {"path": "x"}})
    assert a["tool"] == "read_file" and a["args"]["path"] == "x"
    # alternate key names
    a = agents.normalize_action({"name": "write_file",
                                 "arguments": {"path": "a", "content": "hi"}})
    assert a["tool"] == "write_file" and a["args"]["content"] == "hi"
    # OpenAI-style function call with stringified arguments
    a = agents.normalize_action({"function": {"name": "run_bash",
                                              "arguments": '{"command": "ls"}'}})
    assert a["tool"] == "run_bash" and a["args"]["command"] == "ls"
    # args inlined at the top level
    a = agents.normalize_action({"tool": "write_file", "path": "a", "content": "x"})
    assert a["args"]["path"] == "a" and a["args"]["content"] == "x"
    a = agents.normalize_action({"tool": "consult_models", "prompt": "advise",
                                 "max_models": 3})
    assert a["args"]["prompt"] == "advise" and a["args"]["max_models"] == 3
    a = agents.normalize_action({"tool": "master_audio", "path": "in.wav",
                                 "output": "out.wav", "profile": "tunecore"})
    assert a["args"]["output"] == "out.wav" and a["args"]["profile"] == "tunecore"
    assert agents.normalize_action({"thought": "no tool here"}) is None


def _check_provider_failover():
    settings = Settings(api_key="x", min_request_interval=0.0)
    reg = ModelRegistry(settings)
    client = _FailProviderClient()
    reg.load(client)
    router = ModelRouter(settings, client, reg)
    text, model = router.call("reasoning", [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    # The broken fireworks model is top-ranked and tried first; the router must
    # still recover by routing to the working OpenRouter model.
    assert not model.startswith("fireworks:"), model
    assert text.startswith("ok from"), text
    assert client.request_count == 2, client.request_count


def _check_response_cache():
    settings = Settings(api_key="x", min_request_interval=0.0, use_cache=True)
    reg = ModelRegistry(settings)
    client = FakeClient()
    reg.load(client)
    router = ModelRouter(settings, client, reg)
    msgs = [{"role": "system", "content": "integrator"},
            {"role": "user", "content": "same request"}]
    t1, m1 = router.call("general", msgs, temperature=0.1)  # cacheable
    t2, m2 = router.call("general", msgs, temperature=0.1)  # should hit cache
    assert t1 == t2, (t1, t2)
    assert client.request_count == 1, "cache did not prevent a 2nd API call"
    assert router.cache_hits == 1 and m2.startswith("cache:"), (router.cache_hits, m2)
    # high-temperature calls must NOT be cached (preserve diversity)
    router.call("general", msgs, temperature=0.9)
    router.call("general", msgs, temperature=0.9)
    assert client.request_count == 3, client.request_count


def _check_godmode_fanout():
    settings = Settings(api_key="x", min_request_interval=0.0,
                        godmode_width=3, max_concurrency=3)
    client = FakeClient()
    reg = ModelRegistry(settings)
    reg.load(client)
    router = ModelRouter(settings, client, reg)
    result = GodMode(settings, router, reg).run(
        "Compare three options.", role="general", width=3, synthesize=True)
    assert len(result.answers) == 3, result.answers
    assert len(result.successful) == 3, result.answers
    assert result.synthesis, "missing synthesis"
    assert result.judge_model, "missing synthesis model"
    report = format_report(result)
    assert "GodMode Report" in report and "Model answers" in report, report


def _check_godmode_all_models_and_auth_isolation():
    settings = Settings(api_key="x", min_request_interval=0.0,
                        godmode_width=3, max_concurrency=3)
    all_client = _GodModeAllClient()
    reg = ModelRegistry(settings)
    reg.load(all_client)
    router = ModelRouter(settings, all_client, reg)
    result = GodMode(settings, router, reg).run(
        "ask all", role="all", width=3, synthesize=False)
    assert len(result.answers) == 3, [a.model_id for a in result.answers]
    assert len(result.successful) == 3, result.answers

    auth_settings = Settings(api_key="x", min_request_interval=0.0,
                             godmode_width=3, max_concurrency=1)
    auth_client = _AuthFailThenOkClient()
    reg = ModelRegistry(auth_settings)
    reg.load(auth_client)
    router = ModelRouter(auth_settings, auth_client, reg)
    result = GodMode(auth_settings, router, reg).run(
        "keep going", role="all", width=3, synthesize=False)
    assert len(result.answers) == 3, result.answers
    assert len(result.successful) == 1, result.answers
    assert any("auth failed" in a.error for a in result.answers if not a.ok), result.answers
    assert any("skipped" in a.error for a in result.answers if not a.ok), result.answers

    text, model = router.call("general", [
        {"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}])
    assert text.startswith("ok from"), (text, model)
    assert not model.startswith("huggingface:"), model


def _check_undo():
    root = tempfile.mkdtemp(prefix="ofo_undo_")
    try:
        tb = ToolBox(Settings(api_key="x"), _SilentUI(), root=root,
                     auto_approve=True, allow_bash=False)
        tb.run("write_file", {"path": "f.txt", "content": "v1"})
        tb.run("write_file", {"path": "f.txt", "content": "v2"})
        assert open(os.path.join(root, "f.txt")).read() == "v2"
        assert tb.undo().ok                       # back to v1
        assert open(os.path.join(root, "f.txt")).read() == "v1"
        assert tb.undo().ok                       # undo the create -> removed
        assert not os.path.exists(os.path.join(root, "f.txt"))
        assert not tb.undo().ok                   # nothing left
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _check_anti_stall():
    """A model that keeps trying to read the same file must NOT loop forever."""
    root = tempfile.mkdtemp(prefix="ofo_stall_")
    try:
        with open(os.path.join(root, "a.txt"), "w") as fh:
            fh.write("hello")
        read = json.dumps({"thought": "inspect", "tool": "read_file",
                           "args": {"path": "a.txt"}})
        settings = Settings(api_key="x", min_request_interval=0.0,
                            verify_finish=False, max_steps=12)
        client = _ScriptedClient([read] * 10)
        reg = ModelRegistry(settings)
        reg.load(client)
        router = ModelRouter(settings, client, reg)
        tb = ToolBox(settings, _SilentUI(), root=root,
                     auto_approve=True, allow_bash=False)
        agent = Agent(settings, router, _SilentUI(), tb, root=root)
        result = agent.handle("inspect a.txt")
        assert "Stopped" in result, result
        assert client.request_count < settings.max_steps, client.request_count
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _check_leaderboard():
    root = tempfile.mkdtemp(prefix="ofo_lb_")
    try:
        path = os.path.join(root, "lb.json")
        lb = Leaderboard(path)
        for _ in range(5):
            lb.record("good/model:free", ok=True, latency=2.0)
        for _ in range(5):
            lb.record("bad/model:free", ok=False)
        assert lb.bias("good/model:free") > lb.bias("bad/model:free")
        lb.save()
        reloaded = Leaderboard(path)          # persistence round-trip
        assert reloaded.top(1)[0][0] == "good/model:free", reloaded.top(2)
    finally:
        shutil.rmtree(root, ignore_errors=True)


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def run_self_tests(ui) -> int:
    checks = [
        ("json extraction", _check_json_extraction),
        ("plan parsing", _check_plan_parsing),
        ("verdict parsing", _check_verdict_parsing),
        ("registry filter/select", _check_registry),
        ("multi-provider registry", _check_multi_provider_registry),
        ("openrouter zero-price filter", _check_openrouter_catalogue_filter),
        ("godmode api provider adapters", _check_godmode_api_provider_adapters),
        ("godmode provider catalog", _check_godmode_provider_catalog),
        ("model rotation on 429", _check_model_rotation),
        ("provider failover (404 -> working)", _check_provider_failover),
        ("kimi/moonshot provider wired", _check_moonshot_provider),
        ("login key store (add/replace/remove)", _check_login_env_update),
        ("non-chat model filter", _check_non_chat_filter),
        ("temperature/param negotiation", _check_temperature_adaptation),
        ("--free-only excludes paid models", _check_free_only),
        ("lenient json (multiline content)", _check_lenient_json),
        ("action normalization", _check_normalize_action),
        ("response cache", _check_response_cache),
        ("godmode fanout", _check_godmode_fanout),
        ("godmode all models/auth isolation", _check_godmode_all_models_and_auth_isolation),
        ("tools (sandbox/read/write/edit)", _check_tools),
        ("model consultation tool", _check_consult_models_tool),
        ("godmode panel diversity", _check_godmode_diversity),
        ("consult_models default cap", _check_consult_default_cap),
        ("web ui events + approval", _check_web_events),
        ("web server (http/sse/api)", _check_web_server),
        ("pro supporter licenses", _check_pro_licenses),
        ("audio mastering tool", _check_audio_mastering_tool),
        ("file undo", _check_undo),
        ("anti-stall loop guard", _check_anti_stall),
        ("model leaderboard persistence", _check_leaderboard),
        ("agent loop (fake client)", _check_agent_loop),
        ("full pipeline (fake client)", _check_full_pipeline),
    ]
    failures = 0
    ui.rule("Self-tests (offline)")
    for name, fn in checks:
        try:
            fn()
            ui.note("PASS  " + name)
        except AssertionError as exc:
            failures += 1
            ui.error("FAIL  %s: %s" % (name, exc))
        except (OpenRouterError, Exception) as exc:  # noqa: BLE001
            failures += 1
            ui.error("ERROR %s: %s" % (name, exc))
    if failures:
        ui.error("%d/%d self-tests failed." % (failures, len(checks)))
        return 1
    ui.note("All %d self-tests passed." % len(checks))
    return 0
