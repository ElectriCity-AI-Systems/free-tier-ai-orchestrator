"""GodMode-style parallel model fanout.

smol-ai/GodMode broadcasts one prompt into several chat webapps so the user can
compare independent answers. This module ports that core idea to this CLI's API
provider stack: pick diverse healthy models, ask them in parallel, then
optionally synthesize the best single answer.
"""
from __future__ import annotations

import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional

from .client import (AuthError, BudgetExceeded, ModelUnavailable,
                     OpenRouterError, RateLimited, TransientError)
from .registry import ROLE_REASONING, _diversify


@dataclass
class GodModeAnswer:
    model_id: str
    provider: str
    text: str = ""
    ok: bool = False
    error: str = ""
    elapsed: float = 0.0


@dataclass
class GodModeResult:
    prompt: str
    answers: List[GodModeAnswer]
    synthesis: str = ""
    judge_model: str = ""
    synthesis_error: str = ""
    elapsed: float = 0.0

    @property
    def successful(self) -> List[GodModeAnswer]:
        return [a for a in self.answers if a.ok and a.text.strip()]


def _panel_messages(prompt: str) -> List[dict]:
    system = (
        "You are one independent model in a GodMode-style multi-model panel. "
        "Answer the user's prompt directly and completely. Do not mention other "
        "models, panels, or synthesis; provide your best standalone answer."
    )
    return [{"role": "system", "content": system},
            {"role": "user", "content": prompt}]


def _synthesis_messages(prompt: str, answers: List[GodModeAnswer]) -> List[dict]:
    system = (
        "You synthesize a GodMode panel. Multiple models answered the same "
        "prompt independently. Produce one superior final answer: keep correct "
        "details, resolve conflicts, remove duplication, and call out important "
        "uncertainty when the answers disagree. Output only the final answer."
    )
    parts = ["USER PROMPT:\n" + prompt, "MODEL ANSWERS:"]
    for idx, answer in enumerate(answers, 1):
        parts.append("### ANSWER %d - %s\n%s" % (
            idx, answer.model_id, answer.text.strip()))
    parts.append("Now write the single best synthesized answer.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)}]


def format_report(result: GodModeResult) -> str:
    """Markdown report for -o output."""
    lines = ["# GodMode Report", "", "## Prompt", "", result.prompt.strip(), ""]
    if result.synthesis:
        via = " via %s" % result.judge_model if result.judge_model else ""
        lines.extend(["## Synthesis%s" % via, "", result.synthesis.strip(), ""])
    elif result.synthesis_error:
        lines.extend(["## Synthesis", "", "Failed: " + result.synthesis_error, ""])
    lines.extend(["## Model answers", ""])
    for answer in result.answers:
        status = "ok" if answer.ok else "failed"
        lines.append("### %s (%s, %.1fs)" % (answer.model_id, status, answer.elapsed))
        lines.append("")
        lines.append(answer.text.strip() if answer.ok else "Failed: " + answer.error)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


class GodMode:
    def __init__(self, settings, router, registry):
        self.s = settings
        self.router = router
        self.registry = registry
        self._dead_auth_providers = set()
        self._dead_auth_lock = threading.Lock()

    def _call_one(self, model, messages: List[dict]) -> GodModeAnswer:
        started = time.monotonic()
        with self._dead_auth_lock:
            if model.provider in self._dead_auth_providers:
                return GodModeAnswer(
                    model.id, model.provider, "", False,
                    "skipped: provider auth already failed",
                    time.monotonic() - started,
                )
        try:
            text = self.router.client.chat(
                model.id, messages, self.s.temperature, self.s.max_tokens)
            elapsed = time.monotonic() - started
            self.registry.reward(model.id, latency=elapsed)
            self.router.models_used.append(model.id)
            return GodModeAnswer(model.id, model.provider, text, True, "", elapsed)
        except RateLimited as exc:
            self.registry.penalize(model.id, exc.retry_after, hard=True)
            error = "rate-limited"
        except ModelUnavailable as exc:
            self.registry.penalize(model.id, hard=True)
            error = "unavailable: " + str(exc)[:180]
        except TransientError as exc:
            self.registry.penalize(model.id)
            error = "transient error: " + str(exc)[:180]
        except AuthError as exc:
            self.registry.penalize(model.id, hard=True)
            with self._dead_auth_lock:
                self._dead_auth_providers.add(model.provider)
            error = "auth failed: " + str(exc)[:180]
        except BudgetExceeded:
            raise
        except OpenRouterError as exc:
            self.registry.penalize(model.id)
            error = str(exc)[:180]
        return GodModeAnswer(
            model.id, model.provider, "", False, error, time.monotonic() - started)

    def _models_for(self, role: str, width: int):
        role = (role or ROLE_REASONING).strip().lower()
        if role in ("all", "any", "available", "*"):
            pool = [m for m in self.registry.models if self.registry.available(m.id)]
            if not pool:
                pool = self.registry.models
            pool = sorted(pool, key=lambda m: self.registry.score(m, "general"),
                          reverse=True)
            # Distinct model families, not 4 near-identical variants of one model.
            return _diversify(pool, width)
        return self.registry.select(role, n=width)

    def run(self, prompt: str, role: str = ROLE_REASONING, width: Optional[int] = None,
            synthesize: bool = True) -> GodModeResult:
        started = time.monotonic()
        width = max(1, int(width or self.s.godmode_width))
        models = self._models_for(role, width)
        messages = _panel_messages(prompt)
        answers: List[GodModeAnswer] = []

        workers = min(max(1, self.s.max_concurrency), max(1, len(models)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self._call_one, model, messages): idx
                       for idx, model in enumerate(models)}
            ordered = {}
            for fut in as_completed(futures):
                ordered[futures[fut]] = fut.result()
        for idx in sorted(ordered):
            answers.append(ordered[idx])

        result = GodModeResult(
            prompt=prompt, answers=answers, elapsed=time.monotonic() - started)
        good = result.successful
        if synthesize and len(good) == 1:
            result.synthesis = good[0].text
            result.judge_model = "single:" + good[0].model_id
        elif synthesize and len(good) > 1:
            try:
                synthesis, judge_model = self.router.call(
                    ROLE_REASONING, _synthesis_messages(prompt, good),
                    temperature=0.2, max_tokens=self.s.max_tokens * 2)
                result.synthesis = synthesis
                result.judge_model = judge_model
            except (AuthError, BudgetExceeded):
                raise
            except OpenRouterError as exc:
                result.synthesis_error = str(exc)[:220]
        result.elapsed = time.monotonic() - started
        return result
