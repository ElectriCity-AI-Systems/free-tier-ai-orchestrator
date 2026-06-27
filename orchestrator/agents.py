"""Prompt construction and response parsing for each agent role.

Roles are intentionally small and composable:
  planner    -> decomposes the goal into verifiable subtasks
  worker     -> solves one subtask
  judge      -> merges several candidate solutions into one
  critic     -> scores a solution against acceptance criteria
  aggregator -> integrates all subtask results into the final deliverable
  verifier   -> scores the final deliverable against the original goal
  refiner    -> revises the deliverable using verifier feedback

Free models are not reliable JSON emitters, so every parser is defensive:
it tolerates prose around the JSON, code fences, and partial structures, and
always degrades gracefully instead of raising.
"""
from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple, TYPE_CHECKING

from .registry import ROLE_CODER, ROLE_GENERAL, ROLE_REASONING

if TYPE_CHECKING:  # avoid an import cycle; only needed for type hints
    from .state import SubTask

VALID_ROLES = {ROLE_GENERAL, ROLE_CODER, ROLE_REASONING}


# --------------------------------------------------------------------------- #
# JSON extraction helpers
# --------------------------------------------------------------------------- #
def _lenient_loads(snippet: str):
    """json.loads that tolerates the mistakes weak models make.

    Crucially uses strict=False so LITERAL newlines/tabs inside string values
    parse - that is exactly what models emit for multi-line file content, and
    rejecting it was the #1 cause of 'invalid action' stalls.
    """
    try:
        return json.loads(snippet, strict=False)
    except (ValueError, TypeError):
        pass
    # Repair pass: drop trailing commas before } or ].
    repaired = re.sub(r",(\s*[}\]])", r"\1", snippet)
    try:
        return json.loads(repaired, strict=False)
    except (ValueError, TypeError):
        return None


def _balanced_json(text: str):
    """Find and parse the first balanced {...} or [...] in `text`."""
    start = -1
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            break
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for j in range(start, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                depth += 1
            elif ch in "}]":
                depth -= 1
                if depth == 0:
                    return _lenient_loads(text[start:j + 1])
    return None


def extract_json(text: str):
    """Best-effort JSON extraction from a possibly chatty model response."""
    if not text:
        return None
    fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    for candidate in fenced + [text]:
        obj = _lenient_loads(candidate.strip())
        if obj is not None:
            return obj
        obj = _balanced_json(candidate)
        if obj is not None:
            return obj
    return None


# Keys different models use for the tool name and its arguments.
_TOOL_KEYS = ("tool", "name", "action", "tool_name")
_ARG_KEYS = ("args", "arguments", "input", "parameters", "params")
_INLINE_ARGS = (
    "path", "content", "find", "replace", "command", "summary", "answer",
    "prompt", "question", "task", "role", "max_models",
    "output", "profile", "target_lufs", "true_peak", "lra",
    "sample_rate", "bit_depth", "channels",
)


def normalize_action(obj):
    """Coerce many plausible model outputs into {tool, args, thought} or None.

    Handles: plain {"tool","args"}, alternate key names, OpenAI-style
    {"function":{"name","arguments"}}, args inlined at the top level, and a
    list of actions (the first valid one wins).
    """
    if isinstance(obj, list):
        for item in obj:
            action = normalize_action(item)
            if action:
                return action
        return None
    if not isinstance(obj, dict):
        return None

    thought = ""
    for k in ("thought", "reasoning", "thoughts", "plan"):
        if isinstance(obj.get(k), str):
            thought = obj[k].strip()
            break

    # OpenAI-ish function-call shape: {"function": {"name", "arguments"}}
    fn = obj.get("function")
    if isinstance(obj.get("tool"), dict):
        fn = obj["tool"]
    if isinstance(fn, dict):
        tool = fn.get("name") or fn.get("tool")
        raw_args = fn.get("arguments", fn.get("args", {}))
        if isinstance(raw_args, str):
            raw_args = _lenient_loads(raw_args) or {}
        if isinstance(tool, str):
            return {"tool": tool.strip(),
                    "args": raw_args if isinstance(raw_args, dict) else {},
                    "thought": thought}

    tool = None
    for k in _TOOL_KEYS:
        if isinstance(obj.get(k), str):
            tool = obj[k].strip()
            break
    if not tool:
        return None

    args = None
    for k in _ARG_KEYS:
        if isinstance(obj.get(k), dict):
            args = obj[k]
            break
    if args is None:
        args = {k: obj[k] for k in _INLINE_ARGS if k in obj}
    return {"tool": tool, "args": args if isinstance(args, dict) else {},
            "thought": thought}


def _clamp_score(value, default: int = 0) -> int:
    try:
        return max(0, min(100, int(round(float(value)))))
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
PLANNER_SYSTEM = (
    "You are the lead planner of an autonomous team of AI specialists. "
    "Break the user's GOAL into the SMALLEST set of concrete, independently "
    "verifiable subtasks that together FULLY achieve it. Prefer 2-6 subtasks; "
    "use one subtask only for genuinely simple goals. Order them so each "
    "subtask can build on earlier results via depends_on. Choose role per "
    "subtask: 'coder' for code, 'reasoning' for analysis/math/planning, "
    "'general' otherwise. Reply with JSON ONLY, no commentary."
)

_PLAN_TEMPLATE = """GOAL:
{goal}

Return exactly this JSON shape:
{{"subtasks": [
  {{"id": "t1", "description": "<what to do>", "role": "general|coder|reasoning",
    "acceptance": "<objective, checkable criteria for done>", "depends_on": []}}
]}}"""


def planner_messages(goal: str) -> List[dict]:
    return [
        {"role": "system", "content": PLANNER_SYSTEM},
        {"role": "user", "content": _PLAN_TEMPLATE.format(goal=goal)},
    ]


def parse_plan(text: str) -> Optional[List[dict]]:
    """Return a normalised list of subtask dicts, or None if unusable."""
    obj = extract_json(text)
    if isinstance(obj, dict):
        obj = obj.get("subtasks") or obj.get("tasks") or obj.get("plan")
    if not isinstance(obj, list) or not obj:
        return None

    plan: List[dict] = []
    seen_ids = set()
    for idx, item in enumerate(obj, 1):
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description") or item.get("task") or "").strip()
        if not desc:
            continue
        tid = str(item.get("id") or ("t%d" % idx)).strip() or ("t%d" % idx)
        if tid in seen_ids:
            tid = "t%d" % idx
        seen_ids.add(tid)
        role = str(item.get("role", ROLE_GENERAL)).strip().lower()
        if role not in VALID_ROLES:
            role = ROLE_GENERAL
        acceptance = str(item.get("acceptance") or item.get("criteria")
                         or "Fully and correctly satisfies the subtask.").strip()
        deps_raw = item.get("depends_on") or item.get("deps") or []
        deps = [str(d).strip() for d in deps_raw] if isinstance(deps_raw, list) else []
        plan.append({"id": tid, "description": desc, "role": role,
                     "acceptance": acceptance, "depends_on": deps})

    # Drop dependencies that reference unknown ids (defensive).
    valid_ids = {p["id"] for p in plan}
    for p in plan:
        p["depends_on"] = [d for d in p["depends_on"] if d in valid_ids and d != p["id"]]
    return plan or None


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
def worker_messages(goal: str, task: "SubTask", dep_context: str = "",
                    feedback: Optional[str] = None) -> List[dict]:
    system = (
        "You are a %s specialist on an autonomous team. Solve ONLY your "
        "assigned subtask, completely and correctly. Be precise, concrete and "
        "self-contained. If you write code, make it runnable." % task.role
    )
    parts = [
        "OVERALL GOAL:\n" + goal,
        "YOUR SUBTASK:\n" + task.description,
        "ACCEPTANCE CRITERIA:\n" + task.acceptance,
    ]
    if dep_context:
        parts.append("RESULTS FROM PRIOR SUBTASKS (build on these):\n" + dep_context)
    if feedback:
        parts.append("A reviewer rejected your previous attempt. REVISE to "
                     "address this feedback:\n" + feedback)
    parts.append("Deliver the final result for THIS subtask now.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(parts)}]


# --------------------------------------------------------------------------- #
# Judge (ensemble synthesis)
# --------------------------------------------------------------------------- #
def judge_messages(task: "SubTask", candidates: List[str]) -> List[dict]:
    system = (
        "You are an expert judge. Several specialists independently solved the "
        "same subtask. Produce ONE superior solution: keep what is correct, fix "
        "errors, and merge complementary strengths. Output only the final "
        "merged solution - no meta commentary about the candidates."
    )
    body = ["SUBTASK:\n" + task.description, "ACCEPTANCE:\n" + task.acceptance]
    for i, cand in enumerate(candidates, 1):
        body.append("--- CANDIDATE %d ---\n%s" % (i, cand))
    body.append("Now write the single best solution.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(body)}]


# --------------------------------------------------------------------------- #
# Critic
# --------------------------------------------------------------------------- #
CRITIC_SYSTEM = (
    "You are a rigorous, skeptical reviewer. Judge whether the SOLUTION fully "
    "meets the ACCEPTANCE CRITERIA for the subtask. Be strict: partial or "
    "plausible-but-unverified work does not pass. Reply with JSON ONLY: "
    '{"score": <0-100>, "passed": <true|false>, "feedback": "<specific, '
    'actionable fixes>"}'
)


def critic_messages(task: "SubTask", solution: str) -> List[dict]:
    user = (
        "SUBTASK:\n%s\n\nACCEPTANCE CRITERIA:\n%s\n\nSOLUTION:\n%s\n\n"
        "Return the JSON verdict."
        % (task.description, task.acceptance, solution)
    )
    return [{"role": "system", "content": CRITIC_SYSTEM},
            {"role": "user", "content": user}]


def parse_verdict(text: str, threshold: int) -> Tuple[int, bool, str]:
    """Return (score, passed, feedback). Conservative on parse failure."""
    obj = extract_json(text)
    if not isinstance(obj, dict):
        # No structured verdict: treat as not-yet-passing but not catastrophic.
        return 60, False, (text or "Reviewer returned no structured verdict.")[:500]
    score = _clamp_score(obj.get("score"), default=0)
    feedback = str(obj.get("feedback") or obj.get("issues") or "").strip()
    passed = obj.get("passed")
    if not isinstance(passed, bool):
        passed = score >= threshold
    # Require BOTH the explicit flag and the threshold to avoid false passes.
    passed = bool(passed) and score >= threshold
    return score, passed, feedback


# --------------------------------------------------------------------------- #
# Aggregator
# --------------------------------------------------------------------------- #
def aggregator_messages(goal: str, results: List[Tuple["SubTask", str]]) -> List[dict]:
    system = (
        "You are the integrator. Combine the subtask results into ONE cohesive, "
        "complete deliverable that fully satisfies the user's goal. Resolve "
        "conflicts, remove redundancy, keep everything consistent, and present "
        "it clearly and ready to use."
    )
    body = ["USER GOAL:\n" + goal, "SUBTASK RESULTS:"]
    for task, result in results:
        body.append("### %s - %s\n%s" % (task.id, task.description, result))
    body.append("Produce the final, integrated deliverable now.")
    return [{"role": "system", "content": system},
            {"role": "user", "content": "\n\n".join(body)}]


# --------------------------------------------------------------------------- #
# Final verifier + refiner
# --------------------------------------------------------------------------- #
VERIFIER_SYSTEM = (
    "You verify whether a DELIVERABLE fully achieves the user's GOAL. Be "
    "objective. Reply with JSON ONLY: "
    '{"score": <0-100>, "complete": <true|false>, '
    '"missing": "<what is missing or wrong; empty string if none>"}'
)


def final_verifier_messages(goal: str, deliverable: str) -> List[dict]:
    user = ("GOAL:\n%s\n\nDELIVERABLE:\n%s\n\nReturn the JSON verdict."
            % (goal, deliverable))
    return [{"role": "system", "content": VERIFIER_SYSTEM},
            {"role": "user", "content": user}]


def parse_final(text: str) -> Tuple[int, bool, str]:
    obj = extract_json(text)
    if not isinstance(obj, dict):
        return 60, False, "Verifier returned no structured verdict."
    score = _clamp_score(obj.get("score"), default=0)
    complete = obj.get("complete")
    if not isinstance(complete, bool):
        complete = score >= 90
    missing = str(obj.get("missing") or "").strip()
    return score, bool(complete), missing


def refine_messages(goal: str, deliverable: str, feedback: str) -> List[dict]:
    system = (
        "Revise the DELIVERABLE so it fully satisfies the GOAL, specifically "
        "addressing the reviewer's notes. Preserve everything already correct. "
        "Output only the improved deliverable."
    )
    user = ("GOAL:\n%s\n\nREVIEWER NOTES:\n%s\n\nCURRENT DELIVERABLE:\n%s\n\n"
            "Write the improved deliverable now." % (goal, feedback, deliverable))
    return [{"role": "system", "content": system},
            {"role": "user", "content": user}]
