"""Model routing: pick the best healthy free model for a role and rotate on
failure. Shared by both the agent loop and the batch pipeline so the
"autonomous, safe model selection" logic lives in exactly one place.

Adds two efficiency wins:
  * a deterministic response cache that dedupes identical low-temperature calls
    (verifier/critic/judge), cutting free-tier requests, and
  * latency tracking that feeds the cross-session leaderboard so faster models
    bubble up over time.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

from .client import (AuthError, BudgetExceeded, ModelUnavailable,
                     OpenRouterError, RateLimited, TransientError)
from .config import Settings
from .registry import ModelRegistry

# Distinct models to try for one logical call before giving up. Higher than it
# looks necessary on purpose: with several providers, a couple may be broken
# (bad key, 404s, throttling) and we want to fall through to the working ones.
MAX_MODEL_TRIES = 8


def _cache_key(messages: List[dict], temperature: float, max_tokens: int) -> str:
    blob = json.dumps([messages, round(temperature, 3), max_tokens],
                      sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ModelRouter:
    def __init__(self, settings: Settings, client, registry: ModelRegistry):
        self.s = settings
        self.client = client
        self.registry = registry
        self.models_used: List[str] = []
        self.cache_hits = 0
        self._cache: "OrderedDict[str, Tuple[str, str]]" = OrderedDict()

    # -- cache helpers ----------------------------------------------------- #
    def _cacheable(self, temperature: Optional[float]) -> bool:
        eff = self.s.temperature if temperature is None else temperature
        return self.s.use_cache and eff <= self.s.cache_temp_threshold

    def _cache_get(self, key: str):
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            self.cache_hits += 1
        return hit

    def _cache_put(self, key: str, value: Tuple[str, str]) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.s.cache_max_entries:
            self._cache.popitem(last=False)

    # -- main call --------------------------------------------------------- #
    def call(self, role: str, messages: List[dict],
             tried: Optional[set] = None,
             temperature: Optional[float] = None,
             max_tokens: Optional[int] = None) -> Tuple[str, str]:
        """Return (text, model_id). Rotates models on rate-limit/error.

        Raises BudgetExceeded immediately (fatal); raises OpenRouterError only
        if every candidate fails. Provider-specific AuthError is treated as a
        hard candidate/vendor failure so one bad key cannot stall other providers.
        """
        cacheable = self._cacheable(temperature)
        key = None
        if cacheable:
            key = _cache_key(messages, self.s.temperature if temperature is None
                             else temperature, max_tokens or self.s.max_tokens)
            hit = self._cache_get(key)
            if hit is not None:
                return hit[0], "cache:" + hit[1]

        tried = set(tried or [])
        dead_vendors: set = set()  # provider/family groups that just failed
        errors: List[str] = []
        for _ in range(MAX_MODEL_TRIES):
            candidates = self.registry.select(role, n=1, exclude=tried,
                                              exclude_vendors=dead_vendors)
            if not candidates:
                break
            model = candidates[0]
            tried.add(model.id)
            started = time.monotonic()
            try:
                text = self.client.chat(model.id, messages, temperature, max_tokens)
                self.registry.reward(model.id, latency=time.monotonic() - started)
                self.models_used.append(model.id)
                if key is not None:
                    self._cache_put(key, (text, model.id))
                return text, model.id
            except RateLimited as exc:
                self.registry.penalize(model.id, exc.retry_after, hard=True)
                dead_vendors.add(model.vendor)
                errors.append("%s: rate-limited" % model.id)
            except ModelUnavailable:
                # 404/400: this model is broken; bench it and skip its whole group.
                self.registry.penalize(model.id, hard=True)
                dead_vendors.add(model.vendor)
                errors.append("%s: unavailable" % model.id)
            except TransientError as exc:
                self.registry.penalize(model.id)
                dead_vendors.add(model.vendor)
                errors.append("%s: %s" % (model.id, str(exc)[:60]))
            except AuthError as exc:
                self.registry.penalize(model.id, hard=True)
                dead_vendors.add(model.vendor)
                errors.append("%s: auth failed: %s" % (model.id, str(exc)[:60]))
            except BudgetExceeded:
                raise
            except OpenRouterError as exc:
                self.registry.penalize(model.id)
                dead_vendors.add(model.vendor)
                errors.append("%s: %s" % (model.id, str(exc)[:60]))
        raise OpenRouterError("All candidate models failed for role '%s': %s"
                              % (role, "; ".join(errors[-MAX_MODEL_TRIES:])))

    def safe_call(self, model_id: str, messages: List[dict]) -> Optional[str]:
        """Best-effort single call (used inside ensembles); returns None on
        non-fatal failure, re-raises only fatal budget errors. Not cached -
        ensembles deliberately want diverse, independent answers."""
        started = time.monotonic()
        try:
            text = self.client.chat(model_id, messages)
            self.registry.reward(model_id, latency=time.monotonic() - started)
            self.models_used.append(model_id)
            return text
        except RateLimited as exc:
            self.registry.penalize(model_id, exc.retry_after, hard=True)
        except AuthError:
            self.registry.penalize(model_id, hard=True)
        except BudgetExceeded:
            raise
        except OpenRouterError:
            self.registry.penalize(model_id)
        return None
