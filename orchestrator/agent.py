"""The interactive agent: a ReAct-style tool-use loop.

Instead of relying on native function-calling (which most free models do not
support well), the agent uses a strict JSON-action protocol that any text
model can follow:

    {"thought": "...", "tool": "read_file", "args": {"path": "x.py"}}

Parsing is lenient (literal newlines in file content are fine) and the loop is
hardened against the failure modes weak models fall into - re-reading the same
file forever, repeating a dead action, or never making a concrete change. When
it cannot continue it returns a useful partial summary instead of giving up
silently. Before it is allowed to `finish`, a *different* model verifies the
work against the goal.
"""
from __future__ import annotations

import os
import time
from typing import List, Optional, Tuple

from . import agents
from .client import AuthError, BudgetExceeded, OpenRouterError
from .config import Settings
from .router import ModelRouter

_SYSTEM_TEMPLATE = """You are an autonomous engineering agent working in a terminal.
Working directory: {cwd}

You reach the user's goal by taking ONE action at a time: choose a tool, observe \
its result, then decide the next action. Keep going until the goal is fully done.

Reply with EXACTLY ONE JSON object and NOTHING else (no markdown, no prose around it):
{{"thought": "<one short sentence>", "tool": "<name>", "args": {{<arguments>}}}}

Available tools:
{tools}

Rules:
- Output ONE json action per turn. Never combine actions or add commentary.
- Literal newlines inside "content" are fine - write real multi-line files.
- Inspect before you change, but do NOT re-read a file you already read.
- Make minimal, correct changes; verify (e.g. run tests) when possible.
- If the user asks for all available models, model collaboration, consensus, or
  multi-model advice, call consult_models before making irreversible choices.
- For audio mastering or release-prep requests, use master_audio on an existing
  audio file; if consult_models is available, consult the models first for
  release/mastering strategy, then create the audio file.
- Paths are relative to the working directory.
- Call finish ONLY when the goal is genuinely and verifiably complete."""


class Agent:
    def __init__(self, settings: Settings, router: ModelRouter, ui, toolbox,
                 root: Optional[str] = None):
        self.s = settings
        self.router = router
        self.ui = ui
        self.toolbox = toolbox
        self.root = root or os.getcwd()
        self.messages: List[dict] = [{"role": "system", "content": self._system()}]

    def _system(self) -> str:
        return _SYSTEM_TEMPLATE.format(cwd=self.root, tools=self.toolbox.spec())

    def reset(self) -> None:
        self.messages = [{"role": "system", "content": self._system()}]
        self.ui.note("Conversation reset.")

    # -- context window management ---------------------------------------- #
    def _context(self) -> List[dict]:
        keep = 16
        if len(self.messages) <= keep + 2:
            return self.messages
        return self.messages[:2] + self.messages[-keep:]

    @staticmethod
    def _sig(tool: str, args: dict) -> str:
        if tool == "run_bash":
            return "run_bash:" + str(args.get("command", ""))
        if tool in ("read_file", "list_dir", "write_file", "edit_file",
                    "master_audio"):
            key = str(args.get("path", ""))
            if tool == "edit_file":
                key += "|" + str(args.get("find", ""))[:40]
            return tool + ":" + key
        if tool == "consult_models":
            return "consult_models:" + str(args.get("prompt", ""))[:80]
        return tool

    def _reanchor(self) -> str:
        tools = ", ".join(sorted(self.toolbox.names()))
        return ('Your last message was not a single valid JSON action. Reply with '
                'EXACTLY ONE JSON object, nothing else. Example: '
                '{"thought":"add a helper","tool":"write_file","args":'
                '{"path":"u.py","content":"def f():\\n    return 1"}}. '
                'Newlines inside content are allowed. Valid tools: ' + tools)

    def _graceful_end(self, log: dict, step: int, reason: str) -> str:
        parts = []
        if log["wrote"]:
            parts.append("created/updated " + ", ".join(sorted(set(log["wrote"]))))
        if log["edited"]:
            parts.append("edited " + ", ".join(sorted(set(log["edited"]))))
        if log.get("audio"):
            parts.append("mastered audio " + ", ".join(sorted(set(log["audio"]))))
        if log.get("consulted"):
            parts.append("consulted models %d time(s)" % len(log["consulted"]))
        if log["bash"]:
            parts.append("ran " + "; ".join(log["bash"][-3:]))
        if log["read"]:
            parts.append("inspected %d file(s)" % len(log["read"]))
        did = "; ".join(parts) if parts else "no changes were made"
        last = log["thoughts"][-1] if log["thoughts"] else ""
        tail = (" Last intent: " + last) if last else ""
        return "Stopped after %d steps (%s). Work done: %s.%s" % (step, reason, did, tail)

    def _show_usage(self, req0: int, hit0: int, mdl0: int, t0: float) -> None:
        reqs = self.router.client.request_count - req0
        hits = self.router.cache_hits - hit0
        models = self.router.models_used[mdl0:]
        self.ui.agent_usage(reqs, hits, models, time.monotonic() - t0)

    # -- main loop --------------------------------------------------------- #
    def handle(self, user_input: str) -> str:
        self.messages.append({"role": "user", "content": user_input})
        goal = user_input
        req0 = self.router.client.request_count
        hit0 = self.router.cache_hits
        mdl0 = len(self.router.models_used)
        t0 = time.monotonic()

        log = {"read": set(), "wrote": [], "edited": [], "audio": [],
               "consulted": [],
               "bash": [], "thoughts": []}
        sig_counts: dict = {}
        read_seen: dict = {}     # abs_path -> mtime
        dir_listed: set = set()  # abs dir paths already listed
        steps_since_change = 0
        verified_once = False
        invalid = 0
        nudged_progress = False
        result = None

        try:
            for step in range(1, self.s.max_steps + 1):
                try:
                    text, model = self.router.call(
                        self.s.agent_role, self._context(),
                        temperature=0.3, max_tokens=self.s.max_tokens)
                except (AuthError, BudgetExceeded):
                    raise
                except OpenRouterError as exc:
                    self.ui.error("All models failed: %s" % exc)
                    result = self._graceful_end(log, step, "no model could respond")
                    break

                action = agents.normalize_action(agents.extract_json(text))
                if not action:
                    invalid += 1
                    self.messages.append({"role": "assistant", "content": text})
                    if invalid >= 4:
                        self.ui.error("Model would not produce a valid action; stopping.")
                        result = self._graceful_end(log, step, "no valid actions")
                        break
                    self.messages.append({"role": "user", "content": self._reanchor()})
                    continue
                invalid = 0

                tool = action["tool"]
                args = action["args"] if isinstance(action["args"], dict) else {}
                thought = action["thought"]
                self.messages.append({"role": "assistant", "content": text})
                self.ui.agent_step(step, model, thought, tool, args)
                if thought:
                    log["thoughts"].append(thought)

                if tool == "finish":
                    summary = str(args.get("summary") or args.get("answer")
                                  or thought or "Done.")
                    if self.s.verify_finish and not verified_once:
                        verified_once = True
                        ok, missing = self._verify(goal, summary)
                        if not ok and step < self.s.max_steps:
                            self.ui.note("Reviewer: not complete yet - continuing.")
                            self.messages.append({"role": "user", "content":
                                "A reviewer judged the goal NOT fully met: %s\n"
                                "Keep working with tools; do not finish until resolved."
                                % (missing or "unspecified gaps")})
                            continue
                    self.ui.agent_finish(summary)
                    result = summary
                    break

                sig = self._sig(tool, args)
                sig_counts[sig] = sig_counts.get(sig, 0) + 1
                if sig_counts[sig] >= 5:
                    self.ui.warn("Repeating-action loop detected; stopping.")
                    result = self._graceful_end(log, step, "stuck repeating one action")
                    break

                # ---- read-cache: refuse to re-read unchanged files -------- #
                short = None
                st = None
                if tool == "read_file":
                    st = self.toolbox.stat(args.get("path", ""))
                    if st and read_seen.get(st[0]) == st[1]:
                        short = ("cached: you already read %s and it is unchanged. "
                                 "Do NOT read it again - make a change or finish."
                                 % args.get("path", ""))
                elif tool == "list_dir":
                    st = self.toolbox.stat(args.get("path", "."))
                    key = st[0] if st else args.get("path", ".")
                    if key in dir_listed and sig_counts[sig] >= 2:
                        short = ("cached: you already listed this directory. Read a "
                                 "file, make a change, or finish.")
                if short is not None:
                    self.ui.agent_observation(tool, True, "(cached - not re-run)")
                    self.messages.append({"role": "user", "content":
                                          "OBSERVATION [%s] %s" % (tool, short)})
                    steps_since_change += 1
                    continue

                # ---- execute --------------------------------------------- #
                if tool not in self.toolbox.names():
                    obs = "unknown tool '%s'; valid tools: %s" % (
                        tool, ", ".join(sorted(self.toolbox.names())))
                    ok = False
                else:
                    res = self.toolbox.run(tool, args)
                    obs, ok = res.output, res.ok

                # ---- bookkeeping ----------------------------------------- #
                if ok and tool == "read_file" and st:
                    read_seen[st[0]] = st[1]
                    log["read"].add(args.get("path", ""))
                elif ok and tool == "list_dir" and st:
                    dir_listed.add(st[0])
                elif ok and tool in ("write_file", "edit_file"):
                    steps_since_change = 0
                    nudged_progress = False
                    stp = self.toolbox.stat(args.get("path", ""))
                    if stp:
                        read_seen.pop(stp[0], None)  # allow a fresh re-read after change
                    dir_listed.clear()
                    (log["wrote"] if tool == "write_file" else log["edited"]).append(
                        args.get("path", ""))
                elif ok and tool == "master_audio":
                    steps_since_change = 0
                    nudged_progress = False
                    dir_listed.clear()
                    log["audio"].append(str(args.get("output") or args.get("path", "")))
                elif ok and tool == "consult_models":
                    steps_since_change = 0
                    nudged_progress = False
                    log["consulted"].append(str(args.get("prompt", ""))[:80])
                elif ok and tool == "run_bash":
                    steps_since_change = 0
                    nudged_progress = False
                    log["bash"].append(str(args.get("command", ""))[:60])
                else:
                    steps_since_change += 1

                # ---- nudges ---------------------------------------------- #
                nudge = ""
                if sig_counts[sig] >= 3:
                    nudge += ("\n[note] You repeated this action %d times with no new "
                              "result - do something DIFFERENT or finish."
                              % sig_counts[sig])
                if steps_since_change >= 5 and not nudged_progress:
                    nudged_progress = True
                    nudge += ("\n[note] Several steps without any change. Make a "
                              "concrete edit/write now, or finish.")

                self.ui.agent_observation(tool, ok, obs)
                self.messages.append({"role": "user", "content":
                    "OBSERVATION [%s] %s:\n%s%s"
                    % (tool, "ok" if ok else "ERROR", obs, nudge)})

            if result is None:
                self.ui.warn("Reached step limit (%d)." % self.s.max_steps)
                result = self._graceful_end(log, self.s.max_steps, "reached step limit")
        finally:
            self._show_usage(req0, hit0, mdl0, t0)
        return result

    # -- cross-model verification ----------------------------------------- #
    def _verify(self, goal: str, summary: str) -> Tuple[bool, str]:
        try:
            text, _ = self.router.call(
                "reasoning", agents.final_verifier_messages(goal, summary),
                temperature=0.1, max_tokens=500)
        except (AuthError, BudgetExceeded):
            raise
        except OpenRouterError:
            return True, ""  # no reviewer available -> don't block finishing
        score, complete, missing = agents.parse_final(text)
        return (complete and score >= self.s.acceptance_threshold), missing
