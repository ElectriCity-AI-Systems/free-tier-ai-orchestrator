"""Terminal UI.

Uses `rich` when it is importable and working; otherwise falls back to plain,
coloured-if-possible text. All output goes through a lock because subtasks run
on worker threads and may report concurrently.
"""
from __future__ import annotations

import sys
import threading
from typing import List, Optional

_RICH = None
try:  # rich is optional and may be broken in some environments
    from rich.console import Console as _RichConsole  # type: ignore
    from rich.panel import Panel as _RichPanel  # type: ignore
    from rich.table import Table as _RichTable  # type: ignore
    _ = _RichConsole().size  # touch it to surface broken installs early
    _RICH = True
except Exception:  # pragma: no cover - depends on environment
    _RICH = False


_ANSI = {
    "reset": "\033[0m", "dim": "\033[2m", "bold": "\033[1m",
    "red": "\033[31m", "green": "\033[32m", "yellow": "\033[33m",
    "blue": "\033[34m", "magenta": "\033[35m", "cyan": "\033[36m",
}


class UI:
    def __init__(self, no_color: bool = False, verbose: bool = False):
        self.verbose = verbose
        self._lock = threading.Lock()
        self._use_rich = bool(_RICH) and not no_color
        self._color = (not no_color) and sys.stdout.isatty()
        self._console = _RichConsole() if self._use_rich else None

    # -- low-level --------------------------------------------------------- #
    def _c(self, text: str, color: str) -> str:
        if not self._color:
            return text
        return _ANSI.get(color, "") + text + _ANSI["reset"]

    def _print(self, text: str = "") -> None:
        with self._lock:
            if self._console is not None:
                self._console.print(text)
            else:
                print(text)
            sys.stdout.flush()

    def rule(self, title: str = "") -> None:
        if self._console is not None:
            with self._lock:
                self._console.rule(title)
        else:
            self._print(self._c("\n== %s ==" % title, "cyan"))

    # -- high-level events ------------------------------------------------- #
    def header(self, goal: str, models_summary: str) -> None:
        if self._console is not None:
            body = "[bold]Goal[/bold]\n%s\n\n[dim]%s[/dim]" % (goal, models_summary)
            with self._lock:
                self._console.print(_RichPanel(body, title="🤖 Free-Tier AI Orchestrator",
                                               border_style="cyan"))
        else:
            self._print(self._c("\n🤖 Free-Tier AI Orchestrator", "bold"))
            self._print(self._c("Goal: ", "bold") + goal)
            self._print(self._c(models_summary, "dim"))

    def show_plan(self, plan) -> None:
        self.rule("Plan")
        if self._console is not None and _RICH:
            table = _RichTable(show_header=True, header_style="bold")
            table.add_column("#", style="cyan", no_wrap=True)
            table.add_column("Role")
            table.add_column("Subtask")
            table.add_column("Depends on", style="dim")
            for t in plan:
                table.add_row(t.id, t.role, t.description,
                              ", ".join(t.depends_on) or "-")
            with self._lock:
                self._console.print(table)
        else:
            for t in plan:
                dep = (" <- " + ", ".join(t.depends_on)) if t.depends_on else ""
                self._print("  %s [%s] %s%s" % (
                    self._c(t.id, "cyan"), t.role, t.description,
                    self._c(dep, "dim")))

    def models_table(self, models, registry, roles) -> None:
        self.rule("Selected free/free-tier models")
        if self._console is not None and _RICH:
            table = _RichTable(show_header=True, header_style="bold")
            table.add_column("Provider", style="magenta")
            table.add_column("Model", style="cyan")
            table.add_column("Ctx", justify="right")
            table.add_column("Tier")
            table.add_column("Tags")
            for m in models:
                table.add_row(m.provider, m.upstream_id or m.id,
                              "%dk" % (m.context_length // 1000),
                              getattr(m, "free_kind", "free"),
                              ", ".join(sorted(m.tags)))
            with self._lock:
                self._console.print(table)
        else:
            for m in models:
                label = "%s:%s" % (m.provider, m.upstream_id or m.id)
                self._print("  %-64s %6dk  %-8s %s" % (
                    label, m.context_length // 1000, getattr(m, "free_kind", "free"),
                    ", ".join(sorted(m.tags))))
        self._print("")
        for role in roles:
            picks = registry.select(role, n=3)
            self._print("  %-10s -> %s" % (
                self._c(role, "magenta"), ", ".join(p.id for p in picks)))

    def task_start(self, task) -> None:
        self._print("%s %s [%s] %s" % (
            self._c("▶", "blue"), self._c(task.id, "cyan"),
            task.role, task.description))

    def task_attempt(self, task, attempt: int, model: str, score: int) -> None:
        col = "green" if score >= 80 else ("yellow" if score >= 50 else "red")
        self._print("   %s try %d via %s -> critic %s" % (
            self._c("·", "dim"), attempt, self._c(model, "dim"),
            self._c("%d/100" % score, col)))

    def task_done(self, task, score: int) -> None:
        self._print("   %s %s accepted (%d/100) via %s" % (
            self._c("✔", "green"), task.id, score, self._c(task.model_used, "dim")))

    def task_partial(self, task, score: int) -> None:
        self._print("   %s %s kept best-effort (%d/100) after %d tries" % (
            self._c("◐", "yellow"), task.id, max(score, 0), task.attempts))

    def task_failed(self, task, error) -> None:
        self._print("   %s %s failed: %s" % (
            self._c("✗", "red"), task.id, str(error)[:200]))

    def note(self, text: str) -> None:
        self._print(self._c("  • " + text, "dim"))

    def warn(self, text: str) -> None:
        self._print(self._c("  ! " + text, "yellow"))

    def error(self, text: str) -> None:
        self._print(self._c("  ✗ " + text, "red"))

    def verify(self, score: int, complete: bool, missing: str) -> None:
        col = "green" if complete else "yellow"
        self._print("%s final verification: %s  complete=%s" % (
            self._c("⚖", "magenta"), self._c("%d/100" % score, col), complete))
        if missing and not complete:
            self.note("missing: " + missing[:300])

    def deliverable(self, text: str) -> None:
        self.rule("Deliverable")
        if self._console is not None and _RICH:
            with self._lock:
                self._console.print(_RichPanel(text, border_style="green"))
        else:
            self._print(text)

    def summary(self, score: int, requests: int, elapsed: float,
                models_used: List[str], status: str) -> None:
        self.rule("Summary")
        col = "green" if score >= 80 else ("yellow" if score >= 50 else "red")
        self._print("  Outcome      : " + self._c(status, col))
        self._print("  Confidence   : " + self._c("%d/100 (critic-assessed)" % score, col))
        self._print("  API requests : %d" % requests)
        self._print("  Wall time    : %.1fs" % elapsed)
        if models_used:
            self._print("  Models used  : " + ", ".join(sorted(set(models_used))[:8]))

    # -- prompts ----------------------------------------------------------- #
    def confirm(self, question: str, assume_yes: bool) -> bool:
        if assume_yes:
            return True
        if not sys.stdin.isatty():
            return False
        with self._lock:
            try:
                ans = input(self._c("? " + question + " [y/N] ", "yellow")).strip().lower()
            except (EOFError, KeyboardInterrupt):
                return False
        return ans in ("y", "yes", "j", "ja")

    # -- agent / REPL rendering ------------------------------------------- #
    def plain(self, text: str) -> None:
        self._print(text)

    def read_prompt(self, label: str) -> str:
        prompt = self._c("\n%s › " % label, "cyan")
        return input(prompt)

    def agent_welcome(self, cwd: str, models_summary: str, auto: bool) -> None:
        body = ("Interactive agent ready. Type a goal; I'll use tools to do it.\n"
                "Working dir: %s\n%s\nauto-approve: %s   (/help for commands)"
                % (cwd, models_summary, "ON" if auto else "OFF"))
        if self._console is not None and _RICH:
            with self._lock:
                self._console.print(_RichPanel(body, title="🤖 Free-Tier AI Agent",
                                               border_style="cyan"))
        else:
            self._print(self._c("\n🤖 Free-Tier AI Agent", "bold"))
            self._print(body)

    @staticmethod
    def _args_preview(tool: str, args: dict) -> str:
        if not isinstance(args, dict):
            return ""
        if "command" in args:
            return str(args["command"])
        if "path" in args:
            return str(args["path"])
        import json as _json
        try:
            return _json.dumps(args)[:120]
        except Exception:
            return str(args)[:120]

    def agent_step(self, step: int, model: str, thought: str,
                   tool: str, args: dict) -> None:
        if thought:
            self._print("%s %s" % (self._c("[%d]" % step, "dim"),
                                   self._c(thought, "dim")))
        self._print("  %s %s %s" % (
            self._c("→", "blue"), self._c(tool, "magenta"),
            self._c(self._args_preview(tool, args), "cyan")))

    def tool_preview(self, title: str, body: str) -> None:
        self._print("    " + self._c(title, "yellow"))
        if body:
            for ln in str(body).splitlines()[:12]:
                self._print("    " + self._c("│ " + ln, "dim"))

    def agent_observation(self, tool: str, ok: bool, output: str) -> None:
        mark = self._c("✔", "green") if ok else self._c("✗", "red")
        first = (output or "").strip().splitlines()
        head = first[0] if first else ""
        extra = (" …(+%d lines)" % (len(first) - 1)) if len(first) > 1 else ""
        self._print("  %s %s" % (mark, self._c((head[:160] + extra) or "(no output)", "dim")))

    def agent_finish(self, summary: str) -> None:
        if self._console is not None and _RICH:
            with self._lock:
                self._console.print(_RichPanel(summary, title="✓ Done",
                                               border_style="green"))
        else:
            self._print(self._c("\n✓ Done:", "green"))
            self._print(summary)

    def agent_usage(self, requests: int, cache_hits: int,
                    models: List[str], elapsed: float) -> None:
        uniq = sorted({m.replace("cache:", "") for m in models})
        extra = ((" · %d cache hit%s" % (cache_hits, "" if cache_hits == 1 else "s"))
                 if cache_hits else "")
        self._print(self._c("  ◷ %d request%s%s · %.1fs · %s" % (
            requests, "" if requests == 1 else "s", extra, elapsed,
            ", ".join(uniq[:4]) or "-"), "dim"))

    def godmode_result(self, result) -> None:
        self.rule("GodMode model answers")
        for answer in result.answers:
            status = "ok" if answer.ok else "failed"
            title = "%s  [%s, %.1fs]" % (answer.model_id, status, answer.elapsed)
            body = answer.text.strip() if answer.ok else ("Failed: " + answer.error)
            if self._console is not None and _RICH:
                style = "green" if answer.ok else "red"
                with self._lock:
                    self._console.print(_RichPanel(body or "(empty)",
                                                   title=title[:120],
                                                   border_style=style))
            else:
                col = "green" if answer.ok else "red"
                self._print(self._c(title, col))
                self._print(body or "(empty)")
                self._print("")

        if result.synthesis:
            label = "GodMode synthesis"
            if result.judge_model:
                label += " via " + result.judge_model
            self.rule(label)
            if self._console is not None and _RICH:
                with self._lock:
                    self._console.print(_RichPanel(result.synthesis.strip(),
                                                   border_style="cyan"))
            else:
                self._print(result.synthesis.strip())
        elif not result.successful:
            self.warn("No GodMode panel answer succeeded.")
        elif getattr(result, "synthesis_error", ""):
            self.warn("Synthesis failed: " + result.synthesis_error)
        else:
            self.note("Synthesis disabled; use the model answers above.")
        self.note("GodMode finished with %d/%d successful answer(s) in %.1fs." % (
            len(result.successful), len(result.answers), result.elapsed))

    def leaderboard_table(self, rows) -> None:
        self.rule("Model leaderboard (learned across sessions)")
        if not rows:
            self.note("no stats yet — run a few tasks and they'll appear here.")
            return
        if self._console is not None and _RICH:
            table = _RichTable(show_header=True, header_style="bold")
            table.add_column("Model", style="cyan")
            table.add_column("ok", justify="right", style="green")
            table.add_column("fail", justify="right", style="red")
            table.add_column("avg s", justify="right")
            for mid, rec in rows:
                table.add_row(mid, str(rec.get("ok", 0)), str(rec.get("fail", 0)),
                              "%.1f" % (rec.get("ema_latency", 0) or 0))
            with self._lock:
                self._console.print(table)
        else:
            for mid, rec in rows:
                self._print("  %-46s ok=%-3d fail=%-3d %.1fs" % (
                    mid, rec.get("ok", 0), rec.get("fail", 0),
                    rec.get("ema_latency", 0) or 0))
