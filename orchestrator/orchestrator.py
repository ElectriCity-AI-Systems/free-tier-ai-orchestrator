"""The autonomous multi-model orchestrator.

Pipeline for one goal:

    plan  ->  execute subtasks (with verify+retry, optionally ensembled)
          ->  aggregate  ->  verify whole deliverable + refine

Every model call goes through `_call_role`, which selects the best healthy
free model for a role and transparently rotates to the next candidate on
rate-limits or failures. That is the heart of "autonomous + safe model
selection": the user never picks a model, and one bad/limited model can never
stall the run.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

from . import agents
from .client import (AuthError, BudgetExceeded, OpenRouterClient, OpenRouterError)
from .config import Settings
from .registry import ROLE_GENERAL, ROLE_REASONING, ModelRegistry
from .router import ModelRouter
from .state import (STATUS_DONE, STATUS_FAILED, STATUS_PARTIAL, SubTask)
from .ui import UI


class Orchestrator:
    def __init__(self, settings: Settings, client: OpenRouterClient,
                 registry: ModelRegistry, ui: UI):
        self.s = settings
        self.client = client
        self.registry = registry
        self.ui = ui
        self.router = ModelRouter(settings, client, registry)
        self.models_used = self.router.models_used  # shared list reference
        self._t0 = 0.0

    # Thin delegators to the shared router (model auto-select + rotation).
    def _call_role(self, role: str, messages: List[dict],
                   tried: Optional[set] = None,
                   temperature: Optional[float] = None,
                   max_tokens: Optional[int] = None) -> Tuple[str, str]:
        return self.router.call(role, messages, tried, temperature, max_tokens)

    def _safe_chat(self, model_id: str, messages: List[dict]) -> Optional[str]:
        return self.router.safe_call(model_id, messages)

    # ------------------------------------------------------------------ #
    # Stage 1: planning
    # ------------------------------------------------------------------ #
    def _plan(self, goal: str) -> List[SubTask]:
        try:
            text, _ = self._call_role(ROLE_REASONING, agents.planner_messages(goal),
                                      max_tokens=1500, temperature=0.3)
            raw = agents.parse_plan(text)
            if raw:
                self.ui.note("Planner produced %d subtask(s)." % len(raw))
                return [SubTask(**item) for item in raw]
        except (AuthError, BudgetExceeded):
            raise
        except OpenRouterError as exc:
            self.ui.warn("Planning failed (%s); using single-task fallback." % str(exc)[:80])

        self.ui.warn("Falling back to a single-task plan.")
        return [SubTask(id="t1", description=goal, role=ROLE_GENERAL,
                        acceptance="Fully and correctly satisfies: " + goal,
                        depends_on=[])]

    # ------------------------------------------------------------------ #
    # Stage 2: execution (parallel where dependencies allow)
    # ------------------------------------------------------------------ #
    def _execute(self, goal: str, plan: List[SubTask]) -> Dict[str, str]:
        results: Dict[str, str] = {}
        done: set = set()
        remaining = list(plan)

        while remaining:
            ready = [t for t in remaining if all(d in done for d in t.depends_on)]
            if not ready:  # broken/cyclic deps - run everything left
                ready = list(remaining)

            snapshot = dict(results)
            workers = min(self.s.max_concurrency, len(ready))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(self._do_task, goal, t, snapshot): t for t in ready}
                for fut in as_completed(futures):
                    task = futures[fut]
                    try:
                        task.result = fut.result()
                    except (AuthError, BudgetExceeded):
                        raise
                    except Exception as exc:  # keep the run alive on one bad task
                        task.status = STATUS_FAILED
                        task.result = "[subtask failed: %s]" % exc
                        self.ui.task_failed(task, exc)
                    done.add(task.id)
                    results[task.id] = task.result

            remaining = [t for t in remaining if t.id not in done]
        return results

    def _do_task(self, goal: str, task: SubTask, dep_snapshot: Dict[str, str]) -> str:
        self.ui.task_start(task)
        dep_context = "\n\n".join(
            "[%s] %s" % (d, dep_snapshot.get(d, "")) for d in task.depends_on
        )
        feedback: Optional[str] = None
        best_text, best_score, best_model = "", -1, ""

        for attempt in range(1, self.s.max_task_attempts + 1):
            task.attempts = attempt

            use_ensemble = (self.s.enable_ensemble and attempt == 1
                            and task.role in (ROLE_REASONING, "coder"))
            if use_ensemble:
                text, model = self._ensemble(goal, task, dep_context)
            else:
                text, model = self._call_role(
                    task.role, agents.worker_messages(goal, task, dep_context, feedback))

            score, passed, fb = self._critique(task, text)
            self.ui.task_attempt(task, attempt, model, score)

            if score > best_score:
                best_text, best_score, best_model = text, score, model
            if passed:
                task.status = STATUS_DONE
                task.result, task.model_used, task.score = text, model, score
                self.ui.task_done(task, score)
                return text
            feedback = fb or "Improve correctness and fully meet the acceptance criteria."

        task.status = STATUS_PARTIAL
        task.result, task.model_used, task.score = best_text, best_model, max(best_score, 0)
        self.ui.task_partial(task, best_score)
        return best_text

    def _ensemble(self, goal: str, task: SubTask, dep_context: str) -> Tuple[str, str]:
        """Race several diverse models, then have a judge merge the best answer."""
        models = self.registry.select(task.role, n=self.s.ensemble_size)
        messages = agents.worker_messages(goal, task, dep_context)
        candidates: List[Tuple[str, str]] = []

        workers = min(self.s.max_concurrency, len(models))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(self._safe_chat, m.id, messages): m for m in models}
            for fut in as_completed(futures):
                model = futures[fut]
                text = fut.result()
                if text:
                    candidates.append((model.id, text))

        if not candidates:
            return self._call_role(task.role, messages)
        if len(candidates) == 1:
            return candidates[0][1], candidates[0][0]

        try:
            merged, judge_model = self._call_role(
                ROLE_REASONING, agents.judge_messages(task, [c[1] for c in candidates]),
                temperature=0.2)
            return merged, "ensemble(%d)+judge:%s" % (len(candidates), judge_model)
        except OpenRouterError:
            # Judge unavailable - fall back to the highest-priority candidate.
            return candidates[0][1], candidates[0][0]

    def _critique(self, task: SubTask, solution: str) -> Tuple[int, bool, str]:
        try:
            text, _ = self._call_role(ROLE_REASONING, agents.critic_messages(task, solution),
                                      max_tokens=700, temperature=0.1)
            return agents.parse_verdict(text, self.s.acceptance_threshold)
        except (AuthError, BudgetExceeded):
            raise
        except OpenRouterError:
            # No reviewer available: don't fake a pass; mark unverified.
            return 65, False, "Reviewer unavailable; result is unverified."

    # ------------------------------------------------------------------ #
    # Stage 3: aggregation
    # ------------------------------------------------------------------ #
    def _aggregate(self, goal: str, plan: List[SubTask], results: Dict[str, str]) -> str:
        pairs = [(t, results.get(t.id, t.result)) for t in plan]
        if len(plan) == 1:
            return pairs[0][1]
        try:
            text, _ = self._call_role(ROLE_GENERAL, agents.aggregator_messages(goal, pairs),
                                      max_tokens=self.s.max_tokens * 2, temperature=0.3)
            return text
        except (AuthError, BudgetExceeded):
            raise
        except OpenRouterError:
            self.ui.warn("Aggregator unavailable; concatenating subtask results.")
            return "\n\n".join("## %s - %s\n%s" % (t.id, t.description, r)
                               for t, r in pairs)

    # ------------------------------------------------------------------ #
    # Stage 4: end-to-end verification + refinement
    # ------------------------------------------------------------------ #
    def _verify_refine(self, goal: str, deliverable: str) -> Tuple[str, int, str]:
        score, complete, missing = 0, False, ""
        for i in range(self.s.refine_passes + 1):
            try:
                text, _ = self._call_role(
                    ROLE_REASONING, agents.final_verifier_messages(goal, deliverable),
                    max_tokens=700, temperature=0.1)
                score, complete, missing = agents.parse_final(text)
            except (AuthError, BudgetExceeded):
                raise
            except OpenRouterError:
                self.ui.warn("Final verifier unavailable; skipping verification.")
                break

            self.ui.verify(score, complete, missing)
            if complete and score >= self.s.acceptance_threshold:
                break
            if i < self.s.refine_passes:
                try:
                    self.ui.note("Refining the deliverable to close the gap…")
                    deliverable, _ = self._call_role(
                        ROLE_GENERAL, agents.refine_messages(goal, deliverable, missing),
                        max_tokens=self.s.max_tokens * 2, temperature=0.3)
                except (AuthError, BudgetExceeded):
                    raise
                except OpenRouterError:
                    break
        return deliverable, score, missing

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #
    def run(self, goal: str) -> dict:
        self._t0 = time.monotonic()
        self.ui.header(goal, self.registry.summary())

        plan = self._plan(goal)
        self.ui.show_plan(plan)

        self.ui.rule("Execution")
        results = self._execute(goal, plan)

        self.ui.rule("Integration")
        deliverable = self._aggregate(goal, plan, results)

        self.ui.rule("Verification")
        deliverable, score, missing = self._verify_refine(goal, deliverable)

        elapsed = time.monotonic() - self._t0
        failed = sum(1 for t in plan if t.status == STATUS_FAILED)
        partial = sum(1 for t in plan if t.status == STATUS_PARTIAL)
        if score >= self.s.acceptance_threshold and failed == 0 and partial == 0:
            status = "GOAL ACHIEVED"
        elif failed or partial or score < self.s.acceptance_threshold:
            status = "PARTIAL - see notes"
        else:
            status = "COMPLETE"

        return {
            "goal": goal,
            "plan": plan,
            "deliverable": deliverable,
            "score": score,
            "missing": missing,
            "status": status,
            "requests": self.client.request_count,
            "elapsed": elapsed,
            "models_used": list(self.models_used),
        }
