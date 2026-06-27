"""Discovery, ranking, health-tracking and selection of free/free-tier models.

Provider model catalogues are the source of truth for *availability* where a
catalogue exists. The APIs do not expose "quality", so we layer a small curated
table on top to bias selection toward families known to be strong, then fall
back to sensible defaults for anything unknown. This keeps the selector working
even as new free-tier models appear.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Set

from .config import PROVIDER_ORDER, Settings

ROLE_GENERAL = "general"
ROLE_CODER = "coder"
ROLE_REASONING = "reasoning"

# Curated quality priors, matched as substrings against the model id.
# (substring, base_score 0-100, capability tags)
QUALITY_TABLE = [
    ("gpt-5.5-codex", 98, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("gpt-5.5", 97, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("gpt-5.1", 96, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("gpt-5", 95, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("claude-opus-4-8", 97, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("claude-sonnet-4-6", 95, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("claude-haiku-4-5", 88, {ROLE_GENERAL, ROLE_CODER}),
    ("claude-3-5-sonnet", 91, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("deepseek-r1", 95, {ROLE_REASONING, ROLE_GENERAL, ROLE_CODER}),
    ("deepseek-v4", 94, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("deepseek-v3", 92, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("deepseek-chat", 90, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("gpt-4.1", 92, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("gpt-4o", 91, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("qwen3-coder", 91, {ROLE_CODER, ROLE_GENERAL, ROLE_REASONING}),
    ("qwen3.7", 90, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("qwen3", 89, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("qwen-2.5-coder", 88, {ROLE_CODER, ROLE_GENERAL}),
    ("qwen2.5-coder", 88, {ROLE_CODER, ROLE_GENERAL}),
    ("kimi", 88, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("glm-5", 88, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("gpt-oss-120b", 87, {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}),
    ("llama-4", 88, {ROLE_GENERAL, ROLE_REASONING}),
    ("llama-3.3-70b", 86, {ROLE_GENERAL, ROLE_REASONING}),
    ("qwq", 87, {ROLE_REASONING}),
    ("gemini-2.5-flash", 88, {ROLE_GENERAL, ROLE_REASONING, ROLE_CODER}),
    ("gemini-2.0-flash", 86, {ROLE_GENERAL}),
    ("llama-3.3", 85, {ROLE_GENERAL}),
    ("nemotron", 84, {ROLE_GENERAL, ROLE_REASONING}),
    ("sonar-deep-research", 90, {ROLE_GENERAL, ROLE_REASONING}),
    ("sonar-reasoning", 88, {ROLE_GENERAL, ROLE_REASONING}),
    ("sonar-pro", 84, {ROLE_GENERAL, ROLE_REASONING}),
    ("sonar", 80, {ROLE_GENERAL}),
    ("gemini-flash", 84, {ROLE_GENERAL}),
    ("mistral-small", 80, {ROLE_GENERAL, ROLE_CODER}),
    ("gemma-3", 78, {ROLE_GENERAL}),
    ("mistral-nemo", 76, {ROLE_GENERAL}),
    ("phi-3", 70, {ROLE_GENERAL}),
]

DEFAULT_SCORE = 60

# Capability hints derived straight from the model id, so unknown but
# clearly-named models still get routed sensibly.
DERIVED_TAGS = [
    ("coder", ROLE_CODER), ("-code", ROLE_CODER), ("code-", ROLE_CODER),
    ("codex", ROLE_CODER), ("gpt-", ROLE_GENERAL), ("claude", ROLE_GENERAL),
    ("sonnet", ROLE_GENERAL), ("opus", ROLE_REASONING), ("sonar", ROLE_GENERAL),
    ("r1", ROLE_REASONING), ("reason", ROLE_REASONING),
    ("think", ROLE_REASONING), ("qwq", ROLE_REASONING),
    ("instruct", ROLE_GENERAL), ("chat", ROLE_GENERAL), ("-it", ROLE_GENERAL),
]


@dataclass
class ModelInfo:
    id: str
    name: str
    context_length: int
    base_score: int
    tags: Set[str]
    provider: str = "openrouter"
    upstream_id: str = ""
    free_kind: str = "zero"

    @property
    def vendor(self) -> str:
        raw = self.upstream_id or self.id
        parts = raw.split("/")
        family = parts[0]
        if "models" in parts:
            idx = parts.index("models")
            if idx + 1 < len(parts):
                family = parts[idx + 1].split("-", 1)[0]
        elif len(parts) == 1:
            family = raw.split("-", 1)[0]
        return "%s/%s" % (self.provider or "openrouter", family)


@dataclass
class _Health:
    cooldown_until: float = 0.0
    failures: int = 0
    successes: int = 0


def _is_free(model: dict) -> bool:
    if model.get("free_tier"):
        return True
    mid = model.get("id", "")
    if mid.endswith(":free"):
        return True
    pricing = model.get("pricing", {}) or {}
    try:
        prompt = float(pricing.get("prompt", "0") or 0)
        completion = float(pricing.get("completion", "0") or 0)
    except (TypeError, ValueError):
        return False
    return prompt == 0.0 and completion == 0.0


def _base_score(model_id: str) -> int:
    mid = model_id.lower()
    for needle, score, _tags in QUALITY_TABLE:
        if needle in mid:
            return score
    return DEFAULT_SCORE


def _tags_for(model_id: str) -> Set[str]:
    mid = model_id.lower()
    tags: Set[str] = set()
    for needle, _score, table_tags in QUALITY_TABLE:
        if needle in mid:
            tags |= table_tags
    for needle, tag in DERIVED_TAGS:
        if needle in mid:
            tags.add(tag)
    tags.add(ROLE_GENERAL)  # every chat model can attempt general work
    return tags


def _diversify(pool: Sequence[ModelInfo], n: int) -> List[ModelInfo]:
    """Pick up to n models, preferring distinct vendors first for diversity.

    Ensemble quality comes from *different* model families disagreeing, so we
    spread the picks across vendors before doubling up.
    """
    chosen: List[ModelInfo] = []
    seen_vendors: Set[str] = set()
    for m in pool:
        if m.vendor not in seen_vendors:
            chosen.append(m)
            seen_vendors.add(m.vendor)
        if len(chosen) >= n:
            return chosen
    for m in pool:  # backfill if we still need more
        if m not in chosen:
            chosen.append(m)
        if len(chosen) >= n:
            break
    return chosen


class ModelRegistry:
    """Holds the free-model catalogue plus live health state."""

    def __init__(self, settings: Settings):
        self.s = settings
        self._models: List[ModelInfo] = []
        self._health: Dict[str, _Health] = {}
        self._lock = threading.Lock()
        self.leaderboard = None  # optional cross-session stats (stats.Leaderboard)

    # -- loading ----------------------------------------------------------- #
    def load(self, client) -> List[ModelInfo]:
        catalogue = client.list_models()
        free: List[ModelInfo] = []
        for m in catalogue:
            if not _is_free(m):
                continue
            if self.s.free_only and m.get("free_kind", "zero") == "credits":
                continue  # skip credit-billed (paid) models in free-only mode
            mid = m.get("id", "")
            if not mid:
                continue
            ctx = m.get("context_length") or (m.get("top_provider", {}) or {}).get("context_length") or 0
            upstream = m.get("upstream_id") or mid
            prefix = mid.split(":", 1)[0] if ":" in mid else ""
            provider = m.get("provider") or (prefix if prefix in PROVIDER_ORDER else "openrouter")
            free.append(ModelInfo(
                id=mid,
                name=m.get("name", mid),
                context_length=int(ctx or 0),
                base_score=_base_score(upstream),
                tags=_tags_for(upstream),
                provider=provider,
                upstream_id=upstream,
                free_kind=m.get("free_kind", "zero"),
            ))
        if not free:
            raise RuntimeError(
                "No free/free-tier models were discovered. Set at least one provider "
                "API key or check your account/region."
            )
        free.sort(key=lambda mi: (mi.base_score, mi.context_length), reverse=True)
        self._models = free
        return free

    @property
    def models(self) -> List[ModelInfo]:
        return list(self._models)

    def summary(self) -> str:
        providers = sorted({m.provider for m in self._models})
        return "%d free/free-tier models across %d provider(s): %s" % (
            len(self._models), len(providers), ", ".join(providers)
        )

    # -- scoring & health -------------------------------------------------- #
    def score(self, m: ModelInfo, role: str) -> float:
        sc = float(m.base_score)
        sc += min(m.context_length / 10000.0, 12.0)  # up to +12 for big context
        if role in m.tags:
            sc += 8.0
        elif role != ROLE_GENERAL and ROLE_GENERAL in m.tags:
            sc += 2.0
        if m.free_kind == "zero":
            sc += 2.0
        elif m.free_kind == "free_api":
            sc += 1.0
        elif m.free_kind == "local":
            sc += 1.0
        elif m.free_kind == "credits":
            sc -= 1.0  # mild; reliability is handled by per-call vendor failover
        h = self._health.get(m.id)
        if h:
            sc += min(h.successes, 5) * 0.5
            sc -= min(h.failures, 6) * 1.5
        if self.leaderboard is not None:
            sc += self.leaderboard.bias(m.id)  # learned cross-session prior
        return sc

    def available(self, model_id: str) -> bool:
        h = self._health.get(model_id)
        return (h is None) or (h.cooldown_until <= time.monotonic())

    def penalize(self, model_id: str, retry_after: float = None, hard: bool = False) -> None:
        with self._lock:
            h = self._health.setdefault(model_id, _Health())
            h.failures += 1
            cooldown = retry_after if retry_after else self.s.model_cooldown_seconds
            if hard:
                cooldown = max(cooldown, self.s.model_cooldown_seconds) * 2.0
            h.cooldown_until = time.monotonic() + min(cooldown, 600.0)
        if self.leaderboard is not None:
            self.leaderboard.record(model_id, ok=False)

    def reward(self, model_id: str, latency: float = None) -> None:
        with self._lock:
            h = self._health.setdefault(model_id, _Health())
            h.successes += 1
            h.failures = max(0, h.failures - 1)
        if self.leaderboard is not None:
            self.leaderboard.record(model_id, ok=True, latency=latency)

    # -- selection --------------------------------------------------------- #
    def select(self, role: str, n: int = 1, exclude: Sequence[str] = (),
               exclude_vendors: Sequence[str] = ()) -> List[ModelInfo]:
        """Return up to n best models for a role, honouring health/cooldowns.

        `exclude_vendors` lets the router skip whole provider/family groups that
        just failed, so one broken provider can't trap the retry loop.
        """
        ex = set(exclude)
        exv = set(exclude_vendors)
        candidates = [m for m in self._models if m.id not in ex and self.available(m.id)]
        if exv:
            filtered = [m for m in candidates if m.vendor not in exv]
            if filtered:  # only honour the vendor exclusion while options remain
                candidates = filtered
        if not candidates:  # last resort: everything is cooling down
            candidates = [m for m in self._models if m.id not in ex] or self._models[:]

        tagged = [m for m in candidates if role in m.tags]
        general = [m for m in candidates if ROLE_GENERAL in m.tags]
        pool = tagged or general or candidates
        pool = sorted(pool, key=lambda m: self.score(m, role), reverse=True)
        return _diversify(pool, max(1, n))
